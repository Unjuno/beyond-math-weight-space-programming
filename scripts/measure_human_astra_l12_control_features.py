#!/usr/bin/env python3
"""Capture L12 feature statistics in fixed phrase-aligned discovery windows."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'results/human_astra_manual_20261105'
RAW=RUN/'raw/discovery_baseline_v1'; BASE=ROOT/'model-cache/Qwen3.5-0.8B'
TENSOR=BASE/'model.safetensors-00001-of-00001.safetensors'; OUT=RUN/'raw/l12_control_feature_stats_v3'
BASE_HASH='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
LAYER=12
CASES={
 'gsm8k_train_6445':('attempt_002',5120,{
  'early_phrase':[(4341,4351)],
  'late_phrases_pooled':[(4635,4645),(4665,4675),(4695,4705),(4725,4735)]}),
 'math_train_algebra_542':('attempt_002',3840,{
  'early_phrase':[(3031,3042)],
  'late_phrases_pooled':[(3338,3349),(3351,3362),(3364,3375),(3377,3388)]}),
 'gsm8k_train_3185':('attempt_003',3490,{
  'ordinary_check':[(384,896)],'reinterpretation':[(1152,1664)],'answer_transition':[(2944,3456)]})}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 if OUT.exists():raise FileExistsError(f'Refuse to overwrite {OUT}')
 if sha(TENSOR)!=BASE_HASH:raise ValueError('STOP: base hash mismatch')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 torch.set_grad_enabled(False);torch.set_num_threads(6)
 AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('mps')
 module=model.model.language_model.layers[LAYER].mlp.down_proj
 W=module.weight.detach().cpu().to(torch.float64).numpy()
 OUT.mkdir(parents=True);records=[]
 for ident,(attempt,upto,groupspec) in CASES.items():
  rawfile=RAW/attempt/'original.jsonl'; rows=[json.loads(s) for s in rawfile.read_text().splitlines() if json.loads(s).get('id')==ident]
  if len(rows)!=1:raise ValueError(f'Expected one raw row: {ident}')
  row=rows[0]
  if row['base_sha256']!=BASE_HASH:raise ValueError(f'Raw base hash mismatch: {ident}')
  gen=row['generated_token_ids'][:upto]; full=row['prompt_token_ids']+gen; plen=len(row['prompt_token_ids'])
  inp=torch.tensor([full],dtype=torch.long,device='mps'); captured={}
  def prehook(module,args):
   x=args[0]
   if x.ndim!=3 or x.shape[-1]!=W.shape[1]:raise ValueError(f'Unexpected input shape {tuple(x.shape)}')
   for group,spans in groupspec.items():
    parts=[]
    for a,b in spans:
     if not (0<=a<b<=upto):raise ValueError(f'Bad span {ident}/{group}/{a}:{b}')
     parts.append(x[0,plen+a:plen+b].detach().cpu().to(torch.float64).numpy())
    z=np.concatenate(parts,axis=0)
    expected_count=sum(b-a for a,b in spans)
    if z.shape[0]!=expected_count:raise ValueError(f'Count mismatch {ident}/{group}: {z.shape[0]} != {expected_count}')
    captured[group]={'count':int(z.shape[0]),'sum':z.sum(0),'sumsq':(z*z).sum(0),'sumabs':np.abs(z).sum(0)}
  handle=module.register_forward_pre_hook(prehook)
  with torch.inference_mode():model.model.language_model(input_ids=inp,use_cache=False,output_hidden_states=False,return_dict=True)
  handle.remove()
  arrays={'weight_matrix':W}; stats={}
  for group,v in captured.items():
   n=v['count'];mean=v['sum']/n;rms=np.sqrt(v['sumsq']/n);meanabs=v['sumabs']/n
   arrays[group+'__mean']=mean;arrays[group+'__rms']=rms;arrays[group+'__mean_abs']=meanabs
   stats[group]={'count':n,'coordinate_1226_mean':float(mean[1226]),'coordinate_1226_rms':float(rms[1226]),'coordinate_1226_mean_abs':float(meanabs[1226])}
  path=OUT/f'{ident}.npz';np.savez_compressed(path,**arrays)
  records.append({'id':ident,'attempt':attempt,'raw_sha256':sha(rawfile),'groups':stats,'file':path.name,'file_sha256':sha(path)})
  print(json.dumps({'id':ident,'groups':stats},ensure_ascii=False),flush=True)
 manifest={'run_id':'human-astra-manual-20261105-v1','measurement':'L12 down_proj-input per-coordinate sum/sumsq/count in exact-phrase and fixed success windows','base_sha256':BASE_HASH,'script_sha256':sha(Path(__file__)),'layer':LAYER,'weight_shape':list(W.shape),'groups':{k:v[2] for k,v in CASES.items()},'items':records,'raw_vectors_local_only':True,'no_weight_edit':True}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'complete','output':str(OUT),'weight_shape':list(W.shape)}),flush=True)

if __name__=='__main__':main()
