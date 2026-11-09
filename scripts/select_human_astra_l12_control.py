#!/usr/bin/env python3
"""Apply the frozen float64 matched-control rule for L12 input coordinate 1226."""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/human_astra_manual_20261105'
STATS=RUN/'raw/l12_control_feature_stats_v3'
CANDIDATE=1226
GROUPS=[
 ('gsm8k_train_6445','early_phrase'),('gsm8k_train_6445','late_phrases_pooled'),
 ('math_train_algebra_542','early_phrase'),('math_train_algebra_542','late_phrases_pooled'),
 ('gsm8k_train_3185','ordinary_check'),('gsm8k_train_3185','reinterpretation'),('gsm8k_train_3185','answer_transition')]

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 m=json.loads((STATS/'manifest.json').read_text())
 rows={r['id']:r for r in m['items']}
 arrays={ident:np.load(STATS/r['file']) for ident,r in rows.items()}
 W=arrays['gsm8k_train_6445']['weight_matrix'].astype(np.float64,copy=False)
 if W.shape!=(1024,3584):raise ValueError(f'Unexpected W shape {W.shape}')
 if not np.isfinite(W).all():raise ValueError('Non-finite weight')
 norms=np.sqrt(np.sum(W*W,axis=0,dtype=np.float64))
 if not np.isfinite(norms).all() or np.any(norms==0):raise ValueError('Zero/nonfinite weight column norm')
 group_rms=[];counts={}
 for ident,group in GROUPS:
  z=arrays[ident][group+'__rms'].astype(np.float64,copy=False)
  # RMS is computed in float64 from predeclared tokens. Rebuild definition from saved Sumsq is not needed here.
  if z.shape!=(W.shape[1],) or not np.isfinite(z).all():raise ValueError(f'Bad RMS vector {ident}/{group}')
  if z[CANDIDATE]==0:raise ValueError(f'Candidate RMS is zero in {ident}/{group}')
  group_rms.append(z)
  counts[f'{ident}/{group}']=rows[ident]['groups'][group]['count']
 R=np.stack(group_rms,axis=0)
 nc=norms[CANDIDATE]
 norm_ratio=norms/nc
 rms_ratio=R/R[:,CANDIDATE,None]
 # Column-direction cosine, accumulated in float64 in ascending element order.
 dot=np.sum(W*W[:,CANDIDATE:CANDIDATE+1],axis=0,dtype=np.float64)
 cosine=dot/(norms*nc)
 f_norm=(norm_ratio>=0.9)&(norm_ratio<=1.1)
 f_act=np.all((rms_ratio>=0.5)&(rms_ratio<=2.0),axis=0)
 f_cos=np.abs(cosine)<=0.1
 reasons=[];dist=np.full(W.shape[1],np.nan,dtype=np.float64);eligible=np.zeros(W.shape[1],dtype=bool)
 for j in range(W.shape[1]):
  why=[]
  if j==CANDIDATE:why.append('candidate_excluded')
  if not (0.9<=norm_ratio[j]<=1.1):why.append('weight_norm_ratio_outside_[0.9,1.1]')
  if not np.all((rms_ratio[:,j]>=0.5)&(rms_ratio[:,j]<=2.0)):why.append('one_or_more_group_rms_ratio_outside_[0.5,2.0]')
  if not (abs(cosine[j])<=0.1):why.append('absolute_weight_column_cosine_above_0.1')
  if not why:
   eligible[j]=True
   dist[j]=math.log(float(norm_ratio[j]))**2+float(np.mean(np.log(rms_ratio[:,j])**2,dtype=np.float64))
  reasons.append(why)
 candidates=np.flatnonzero(eligible)
 candidates=sorted(candidates,key=lambda j:(float(dist[j]),int(j)))
 diag=[]
 for j in range(W.shape[1]):
  diag.append({'coordinate':j,'column_norm_ratio':float(norm_ratio[j]),'group_rms_ratios':[float(x) for x in rms_ratio[:,j]],'weight_column_cosine':float(cosine[j]),'filters_pass':bool(eligible[j]),'exclusion_reasons':reasons[j],'distance':float(dist[j]) if eligible[j] else None})
 selected=candidates[0] if candidates else None
 result={'run_id':'human-astra-manual-20261105-v1','rule_version':'Astra-L12-control-v1-float64','candidate_coordinate':CANDIDATE,'candidate_column_norm':float(nc),'filter_pass_counts':{'column_norm':int(f_norm.sum()),'all_group_rms':int(f_act.sum()),'weight_column_direction_cosine':int(f_cos.sum()),'norm_and_rms':int((f_norm&f_act).sum()),'norm_and_cosine':int((f_norm&f_cos).sum()),'rms_and_cosine':int((f_act&f_cos).sum()),'all_filters':int(eligible.sum()),'eligible_controls_excluding_candidate':int(eligible.sum())},'groups':[{'id':i,'group':g,'count':int(rows[i]['groups'][g]['count'])} for i,g in GROUPS],
  'eligible_control_count':len(candidates),'selected_control_coordinate':int(selected) if selected is not None else None,'status':'matched_control_selected' if selected is not None else 'matched_control_not_found','selected_distance':float(dist[selected]) if selected is not None else None,
  'selected_diagnostics':diag[selected] if selected is not None else None,'diagnostics_path':'all_coordinate_diagnostics.json','float64':True}
 (STATS/'all_coordinate_diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2)+'\n')
 result['diagnostics_sha256']=sha(STATS/'all_coordinate_diagnostics.json')
 (STATS/'control_selection_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
