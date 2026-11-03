#!/usr/bin/env python3
"""Compare the published base/edit Safetensors checkpoints and verify edit tensor hashes."""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open

EXPECTED={
 "model.language_model.layers.15.mlp.down_proj.weight":"82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53",
 "model.language_model.layers.19.mlp.down_proj.weight":"8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2",
}

def digest(t):
 return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--base',type=Path,required=True); p.add_argument('--edited',type=Path,required=True); p.add_argument('--out',type=Path,default=Path('results/checkpoint_audit.json')); a=p.parse_args()
 with safe_open(a.base,framework='pt',device='cpu') as base, safe_open(a.edited,framework='pt',device='cpu') as edit:
  bk=set(base.keys()); ek=set(edit.keys())
  if bk!=ek: raise ValueError(f'tensor keys differ: base-only={len(bk-ek)} edited-only={len(ek-bk)}')
  changes=[]; total=0; hashes={}
  for name in sorted(bk):
   x=base.get_tensor(name); y=edit.get_tensor(name)
   if x.dtype!=y.dtype or x.shape!=y.shape: raise ValueError(f'metadata mismatch: {name}')
   if name in EXPECTED:
    h=digest(y); hashes[name]=h
    if h!=EXPECTED[name]: raise ValueError(f'published edit hash mismatch for {name}: {h}')
   if not torch.equal(x,y):
    count=int(torch.count_nonzero(x!=y)); changes.append({'name':name,'changed_bf16_values':count,'sha256':digest(y)}); total+=count
  if set(x['name'] for x in changes)!=set(EXPECTED): raise ValueError(f'changed tensor set mismatch: {changes}')
 result={'base_checkpoint':str(a.base),'edited_checkpoint':str(a.edited),'base_revision_expected':'2fc06364715b967f1860aea9cf38778875588b17','tensor_count':len(bk),'same_tensor_keys':True,'edited_tensor_hashes':hashes,'changed_tensors':changes,'changed_bf16_values_total':total,'unchanged_tensor_count':len(bk)-len(changes)}
 a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
