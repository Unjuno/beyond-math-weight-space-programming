#!/usr/bin/env python3
"""Paired 2x2 non-math probe: base/author edit x thinking on/off."""
from __future__ import annotations
import argparse, hashlib, json, re, time
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList

class PresencePenalty(LogitsProcessor):
    def __init__(self, prompt_len: int, penalty: float):
        self.prompt_len, self.penalty = prompt_len, penalty
    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len:]
        if generated.numel():
            for row in range(scores.shape[0]):
                ids = torch.unique(generated[row])
                scores[row, ids] -= self.penalty
        return scores

def score(text, rules):
    clean=text.replace('<|im_end|>','').replace('<|endoftext|>','').strip()
    lines=[x.strip() for x in clean.splitlines() if x.strip()]
    result={}
    for i,r in enumerate(rules):
        typ=r['type']; ok=False
        if typ=='exact_bullet_lines': ok=len(lines)==r['count'] and all(x.startswith('-') for x in lines)
        elif typ=='checkbox_lines': ok=len(lines)==r['count'] and all(x.startswith('- [ ]') for x in lines)
        elif typ=='numbered_lines': ok=len(lines)==r['count'] and all(re.match(rf'^{n}[.)]\s+',x) for n,x in enumerate(lines,1))
        elif typ=='line_count': ok=len(lines)==r['count']
        elif typ=='line_prefixes': ok=len(lines)==len(r['values']) and all(x.startswith(p) for x,p in zip(lines,r['values']))
        elif typ=='sentence_count': ok=len([x for x in re.findall(r'[^.!?]+[.!?]?(?=\s|$)',clean) if x.strip()])==r['count']
        elif typ=='word_count_exact': ok=len(re.findall(r"\b[\w’'-]+\b",clean))==r['count']
        elif typ=='word_count_max': ok=len(re.findall(r"\b[\w’'-]+\b",clean))<=r['count']
        elif typ=='contains_all': ok=all(x.casefold() in clean.casefold() for x in r['values'])
        elif typ=='contains_none': ok=all(x.casefold() not in clean.casefold() for x in r['values'])
        elif typ=='no_char': ok=r['value'] not in clean
        elif typ=='exact_text': ok=clean==r['value']
        elif typ=='json_values':
            try:
                obj=json.loads(clean); ok=set(obj)==set(r['values']) and all(obj.get(k)==v for k,v in r['values'].items())
            except (json.JSONDecodeError,AttributeError): ok=False
        else: raise ValueError(f'unknown check {typ}')
        result[f'{i}:{typ}']=bool(ok)
    return result

def repeated_4gram(ids):
    grams=Counter(tuple(ids[i:i+4]) for i in range(max(0,len(ids)-3)))
    return any(n>=3 for n in grams.values())

def weight_hash(weight):
    bits=weight.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(bits.tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--patch-dir',type=Path,required=True)
    ap.add_argument('--prompts',type=Path,default=Path('configs/nonmath_probe.jsonl'))
    ap.add_argument('--out',type=Path,default=Path('results/raw/nonmath_probe.jsonl'))
    ap.add_argument('--max-new-tokens',type=int,default=2048)
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    tasks=[json.loads(x) for x in args.prompts.read_text().splitlines() if x.strip()]
    if args.limit: tasks=tasks[:args.limit]
    done=set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                old=json.loads(line); done.add((old['id'],old['condition']))
    tok=AutoTokenizer.from_pretrained(args.model)
    model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,low_cpu_mem_usage=True).to('mps').eval()
    model.generation_config.use_cache=True
    args.out.parent.mkdir(parents=True,exist_ok=True)
    start=time.time()
    for edited in (False,True):
        if edited:
            for layer,expected in ((15,'82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53'),(19,'8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2')):
                name=f'model.language_model.layers.{layer}.mlp.down_proj.weight'
                target=torch.from_numpy(np.load(args.patch_dir/f'{name}.npy')).to(device='mps',dtype=torch.bfloat16)
                # Use the canonical checkpoint key (the full multimodal wrapper nests the text model).
                weight=model.get_parameter(name)
                weight.data.copy_(target)
                got=weight_hash(weight)
                if got!=expected: raise RuntimeError(f'edited tensor hash mismatch layer {layer}: {got}')
        for thinking in (True,False):
            for index,task in enumerate(tasks):
                condition=('edited' if edited else 'base')+('_thinking' if thinking else '_no_thinking')
                if (task['id'],condition) in done: continue
                seed=4242+index; torch.manual_seed(seed); torch.mps.manual_seed(seed)
                messages=[{'role':'user','content':task['prompt']}]
                enc=tok.apply_chat_template(messages,add_generation_prompt=True,enable_thinking=thinking,tokenize=True,return_dict=True,return_tensors='pt')
                input_ids=enc['input_ids'].to('mps'); plen=input_ids.shape[-1]; t0=time.time()
                with torch.inference_mode():
                    out=model.generate(input_ids=input_ids,max_new_tokens=args.max_new_tokens,do_sample=True,temperature=.6,top_p=.95,top_k=20,logits_processor=LogitsProcessorList([PresencePenalty(plen,1.5)]),eos_token_id=[tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')],pad_token_id=tok.pad_token_id or tok.eos_token_id)
                generated=out[0,plen:].detach().cpu().tolist(); text=tok.decode(generated,skip_special_tokens=False)
                checks=score(text,task['checks'])
                rec={'id':task['id'],'family':task['family'],'condition':condition,'seed':seed,'thinking':thinking,'input_token_ids':input_ids[0].detach().cpu().tolist(),'generated_token_ids':generated,'output':text,'generated_tokens':len(generated),'cap_hit':len(generated)>=args.max_new_tokens,'ended_with_eos':bool(generated and generated[-1] in {tok.eos_token_id,tok.convert_tokens_to_ids('<|im_end|>')}),'repeat_4gram':repeated_4gram(generated),'checks':checks,'all_checks_pass':bool(checks) and all(checks.values()),'seconds':time.time()-t0}
                with args.out.open('a') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
                done.add((task['id'],condition))
                print(f"{rec['condition']} {task['id']} tokens={len(generated)} pass={rec['all_checks_pass']} sec={rec['seconds']:.1f}",flush=True)
    print(f'completed {4*len(tasks)} generations in {time.time()-start:.1f}s; wrote {args.out}')

if __name__=='__main__': main()
