#!/usr/bin/env python3
"""Token-level, full-prefix observation of Astra-nominated L12 coordinate 1226."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/human_astra_manual_20261105'
RAW=RUN/'raw/discovery_baseline_v1'
BASE=ROOT/'model-cache/Qwen3.5-0.8B'
TENSOR=BASE/'model.safetensors-00001-of-00001.safetensors'
OUT=RUN/'raw/l12_coordinate_timecourse_v1'
BASE_HASH='04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696'
LAYER,COORD=12,1226
CASES={
 'gsm8k_train_6445':('attempt_002',5120,{'transition':(4096,4608),'cycle':(4608,5120)}),
 'math_train_algebra_542':('attempt_002',3840,{'transition':(2816,3328),'cycle':(3328,3840)}),
 'gsm8k_train_3185':('attempt_003',3490,{'ordinary_check':(384,896),'reinterpretation':(1152,1664),'answer_transition':(2944,3456)}),
}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def periodic_occurrences(ids, period, stable_start):
 unit=ids[stable_start:stable_start+period]
 if len(unit)!=period:return unit,[]
 matches=[]
 for i in range(0,stable_start+1):
  if ids[i:i+period]==unit:matches.append(i)
 return unit,matches

def main():
 import torch
 from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
 if OUT.exists():raise FileExistsError(f'Refuse to overwrite {OUT}')
 if sha(TENSOR)!=BASE_HASH:raise ValueError('STOP: base model hash mismatch')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
 torch.set_grad_enabled(False);torch.set_num_threads(6)
 tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('mps')
 mlp=model.model.language_model.layers[LAYER].mlp.down_proj
 weight=mlp.weight.detach().float().cpu().numpy()
 if weight.shape[1]<=COORD:raise ValueError(f'coordinate out of bounds: weight {weight.shape}')
 col=weight[:,COORD]; col_norm=float(np.linalg.norm(col)); norms=np.linalg.norm(weight,axis=0)
 OUT.mkdir(parents=True); records=[]
 for ident,(attempt,upto,regions) in CASES.items():
  rawfile=RAW/attempt/'original.jsonl'; rows=[json.loads(s) for s in rawfile.read_text().splitlines() if json.loads(s).get('id')==ident]
  if len(rows)!=1:raise ValueError(f'Need exactly one row: {ident}')
  row=rows[0]
  if row['base_sha256']!=BASE_HASH:raise ValueError(f'raw base mismatch {ident}')
  gen=row['generated_token_ids'][:upto]; prompt=row['prompt_token_ids']; allids=prompt+gen
  inputs=torch.tensor([allids],dtype=torch.long,device='mps'); pfx=len(prompt)
  captured={}
  def prehook(module,args):
   x=args[0]
   if x.ndim!=3 or x.shape[-1]!=weight.shape[1]:raise ValueError(f'Unexpected down_proj input {tuple(x.shape)}')
   selected=x[0,pfx:pfx+upto].detach().float().cpu().numpy()
   captured['coord']=selected[:,COORD]
   captured['sum']=selected.sum(0);captured['sumsq']=(selected*selected).sum(0)
  handle=mlp.register_forward_pre_hook(prehook)
  with torch.inference_mode():model.model.language_model(input_ids=inputs,use_cache=False,output_hidden_states=False,return_dict=True)
  handle.remove()
  gmean=captured['sum']/upto;grms=np.sqrt(captured['sumsq']/upto)
  stats={}
  for name,(a,b) in regions.items():
   x=captured['coord'][a:b];stats[name]={'interval':[a,b],'mean':float(x.mean()),'std':float(x.std()),'mean_abs':float(np.abs(x).mean()),'rms':float(np.sqrt(np.mean(x*x)))}
  arrays={'generated_token_ids':np.asarray(gen,dtype=np.int64),'coordinate_1226':captured['coord'].astype(np.float32),'down_proj_weight_column_1226':col.astype(np.float32),'down_proj_weight_column_norms':norms.astype(np.float32),'input_feature_mean':gmean.astype(np.float32),'input_feature_rms':grms.astype(np.float32)}
  npz=OUT/f'{ident}.npz';np.savez_compressed(npz,**arrays)
  rowrec={'id':ident,'attempt':attempt,'raw_file_sha256':sha(rawfile),'token_count':upto,'prompt_token_count':pfx,'regions':stats,'npz':npz.name,'npz_sha256':sha(npz)}
  if ident=='gsm8k_train_6445': period,start=30,4513
  elif ident=='math_train_algebra_542':period,start=13,3216
  else:period,start=None,None
  if period:
   unit,positions=periodic_occurrences(gen,period,start)
   rowrec['periodic_unit']={'period':period,'stable_cycle_start':start,'all_exact_unit_occurrences_before_stable_start':positions,'unit_token_ids':unit,'unit_text':tok.decode(unit,skip_special_tokens=False),
      'coordinate_values_by_occurrence':{str(pos):captured['coord'][pos:pos+period].astype(float).tolist() for pos in positions}}
  # A preregistered deterministic comparison/control candidate is not selected here.
  rowrec['coordinate_summary']={'layer':LAYER,'coordinate':COORD,'down_proj_input_width':int(weight.shape[1]),'weight_column_l2_norm':col_norm,'column_norm_percentile':float((norms<col_norm).mean()),'activation_mean_all_prefix':float(captured['coord'].mean()),'activation_rms_all_prefix':float(np.sqrt(np.mean(captured['coord']**2)))}
  records.append(rowrec)
  print(json.dumps({'id':ident,'coordinate':rowrec['coordinate_summary'],'regions':stats,'periodic_unit':{k:v for k,v in rowrec.get('periodic_unit',{}).items() if k not in ('coordinate_values_by_occurrence','unit_token_ids')}},ensure_ascii=False),flush=True)
 manifest={'run_id':'human-astra-manual-20261105-v1','measurement':'full-prefix L12 down_proj input coordinate 1226 timecourse and column norm','candidate_status':'observational_only_no_weight_edit','base_sha256':BASE_HASH,'script_sha256':sha(Path(__file__)),'layer':LAYER,'coordinate':COORD,'weight_shape':list(weight.shape),'column_l2_norm':col_norm,'column_norm_percentile':float((norms<col_norm).mean()),'device':'mps','dtype':'bfloat16','attention':'eager','use_cache':False,'rows':records,'raw_vectors_local_only':True,'claim_boundary':'Post-hoc coordinate selection; same-token phase analyses remain descriptive, not causal.'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':'complete','output':str(OUT),'column_norm':col_norm,'column_norm_percentile':manifest['column_norm_percentile']}),flush=True)

if __name__=='__main__':main()
