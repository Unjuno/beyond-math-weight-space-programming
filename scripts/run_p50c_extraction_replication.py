#!/usr/bin/env python3
"""Repeat the three observed extraction wins with fresh paired RNG seeds."""
from __future__ import annotations

import argparse, hashlib, json, random, time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList


class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_len): self.prompt_len = prompt_len
    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len:]
        for row in range(scores.shape[0]):
            if generated.numel(): scores[row, torch.unique(generated[row])] -= 1.5
        return scores


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True); p.add_argument('--patch-dir',type=Path,required=True)
    p.add_argument('--tasks',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True); p.add_argument('--replicates',type=int,default=3)
    a=p.parse_args()
    if a.out.exists() or a.manifest.exists(): raise FileExistsError('refuse to overwrite evidence')
    doc=json.loads(a.tasks.read_text()); tasks={t['id']:t for t in doc['tasks']}
    ids=['x04','x05','x06']
    if not all(i in tasks for i in ids): raise ValueError('required tasks missing')
    torch.set_grad_enabled(False)
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    model=AutoModelForImageTextToText.from_pretrained(a.model,dtype=torch.bfloat16,low_cpu_mem_usage=True,local_files_only=True).to('mps').eval()
    layers={15:'model.language_model.layers.15.mlp.down_proj.weight',19:'model.language_model.layers.19.mlp.down_proj.weight'}
    original={l:model.get_parameter(n).detach().cpu().clone() for l,n in layers.items()}
    patch={}
    for l,n in layers.items():
        patch[l]=torch.from_numpy(__import__('numpy').load(a.patch_dir/f'{n}.npy'))
        model.get_parameter(n).data.copy_(patch[l].to('mps',dtype=torch.bfloat16))
        expected={15:'82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53',19:'8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2'}[l]
        if tensor_sha(model.get_parameter(n))!=expected: raise RuntimeError(f'patch mismatch layer {l}')
        model.get_parameter(n).data.copy_(original[l].to('mps',dtype=torch.bfloat16))
    rows=[]; started=time.time(); end_turn=tok.convert_tokens_to_ids('<|im_end|>')
    seeds=[20261111,20261112,20261113][:a.replicates]
    for task_id in ids:
      task=tasks[task_id]
      enc=tok.apply_chat_template([{'role':'user','content':task['prompt']}],add_generation_prompt=True,enable_thinking=True,tokenize=True,return_dict=True,return_tensors='pt')
      input_ids=enc['input_ids'].to('mps')
      for rep,seed in enumerate(seeds):
       for arm in ['base','edited']:
        for l,n in layers.items(): model.get_parameter(n).data.copy_((original[l] if arm=='base' else patch[l]).to('mps',dtype=torch.bfloat16))
        torch.manual_seed(seed); torch.mps.manual_seed(seed); t0=time.time()
        with torch.inference_mode():
         output=model.generate(input_ids=input_ids,max_new_tokens=32768,do_sample=True,temperature=.6,top_p=.95,top_k=20,logits_processor=LogitsProcessorList([PresencePenalty(input_ids.shape[-1])]),eos_token_id=[tok.eos_token_id,end_turn],pad_token_id=tok.pad_token_id or tok.eos_token_id)
        gen=output[0,input_ids.shape[-1]:].detach().cpu().tolist(); text=tok.decode(gen,skip_special_tokens=False)
        row={'task_id':task_id,'replicate':rep,'seed':seed,'arm':arm,'input_token_ids':input_ids[0].cpu().tolist(),'generated_token_ids':gen,'output':text,'generated_tokens':len(gen),'cap_hit':len(gen)>=32768,'ended_with_eos':bool(gen and gen[-1] in {tok.eos_token_id,end_turn}),'thinking_closed':'</think>' in text,'seconds':time.time()-t0}
        with a.out.open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); __import__('os').fsync(f.fileno())
        rows.append({k:row[k] for k in ['task_id','replicate','seed','arm','generated_tokens','cap_hit','ended_with_eos','thinking_closed','seconds']})
        print(f"{task_id} rep={rep} {arm} tokens={len(gen)} cap={row['cap_hit']} seconds={row['seconds']:.1f}",flush=True)
    manifest={'protocol':'p50c-extraction-followup-v1','tasks':ids,'replicates':seeds,'base_model_sha256':sha(a.model/'model.safetensors-00001-of-00001.safetensors'),'task_sha256':sha(a.tasks),'patch_hashes':{str(l):sha(a.patch_dir/f'{n}.npy') for l,n in layers.items()},'sampling':{'thinking':True,'temperature':.6,'top_p':.95,'top_k':20,'presence_penalty':1.5,'max_new_tokens':32768},'backend':'Transformers BF16 Apple MPS','acceptance':'paired task success = EOS before cap and exact typed JSON match; report all pairs and do not tune edit','results':rows,'elapsed_seconds':time.time()-started,'raw_sha256':sha(a.out)}
    a.manifest.write_text(json.dumps(manifest,indent=2)+'\n')


def tensor_sha(t): return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
if __name__=='__main__': main()
