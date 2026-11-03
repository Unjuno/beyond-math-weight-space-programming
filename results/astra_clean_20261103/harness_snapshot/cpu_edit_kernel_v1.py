"""Persistent CPU-only direct-weight-edit kernel, separate from forward worker.

Software restricted approximation: offline process; no transformers, optimizers,
autograd, forward model, or network APIs in submitted code. All edits audited.
"""
import ast, hashlib, json, os, time, traceback
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from safetensors import safe_open
torch.set_grad_enabled(False);torch.set_num_threads(6)
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'results/astra_clean_20261103'
SOURCE=safe_open(str(ROOT/'model-cache/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors'),framework='pt',device='cpu')
NAMES=[n for n in SOURCE.keys() if n.startswith('model.language_model.')]
FORBIDDEN={'model.language_model.norm.weight','model.language_model.embed_tokens.weight','lm_head.weight'}
def log(obj):
    with open(RUN/'edit_ledger.jsonl','a') as f:f.write(json.dumps({'time':time.time(),**obj})+'\n');f.flush();os.fsync(f.fileno())
def load_original(name):
    assert name in NAMES
    return SOURCE.get_tensor(name).float()
def load(name):
    p=RUN/'edited'/(name+'.npy')
    return torch.from_numpy(np.load(p)) if p.exists() else load_original(name)
def edited():return sorted(p.stem for p in (RUN/'edited').glob('*.npy'))
def drop_edit(name):
    assert name in NAMES
    p=RUN/'edited'/(name+'.npy')
    if p.exists():p.unlink()
    log({'action':'drop_edit','name':name})
def save_edit(name,tensor):
    assert name in NAMES and name not in FORBIDDEN
    tensor=torch.as_tensor(tensor,device='cpu',dtype=torch.float32).detach();base=load_original(name)
    assert tensor.shape==base.shape and torch.isfinite(tensor).all()
    alpha=torch.sum(tensor*base)/torch.sum(base*base)
    resid=torch.linalg.vector_norm(tensor-alpha*base)/torch.linalg.vector_norm(base)
    assert resid>1e-6, 'Whole-tensor rescaling or no edit is out of scope'
    path=RUN/'edited'/(name+'.npy');np.save(path,tensor.numpy())
    log({'action':'save_edit','name':name,'relative_norm':float(torch.linalg.vector_norm(tensor-base)/torch.linalg.vector_norm(base)),
        'changed_elements':int(torch.count_nonzero(tensor-base)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
env={'np':np,'torch':torch,'NAMES':NAMES,'load':load,'load_original':load_original,'save_edit':save_edit,'drop_edit':drop_edit,'edited':edited}
BAD={'backward','autograd','optim','cuda','mps','subprocess','system','socket','requests','transformers','exec','eval','compile','__import__','open'}
def validate(code):
    assert code.lstrip().startswith('# Hypothesis:'),'First line must record hypothesis'
    for n in ast.walk(ast.parse(code)):
        assert not isinstance(n,(ast.Import,ast.ImportFrom)),'Imports belong to harness setup, not editor'
        if isinstance(n,ast.Attribute):assert n.attr not in BAD and not n.attr.startswith('__')
        if isinstance(n,ast.Name):assert n.id not in BAD and not n.id.startswith('__')
        if isinstance(n,ast.Constant) and isinstance(n.value,str):assert n.value not in ['mps','cuda']
    return code
if __name__=='__main__':
    (RUN/'edit_ready.json').write_text(json.dumps({'pid':os.getpid(),'device':'cpu','grad_enabled':torch.is_grad_enabled()}))
    while True:
        pending=[p for p in sorted((RUN/'edit_requests').glob('*.json')) if not (RUN/'edit_responses'/p.name).exists()]
        if not pending:time.sleep(.5);continue
        p=pending[0];req=json.loads(p.read_text());result={'request':str(p),'time':time.time()}
        try:
            code=validate(req['code']);log({'action':'execute','code':code});exec(code,env)
            result.update(status='success',edited=edited(),result=env.get('result'))
        except Exception:result.update(status='failed',error=traceback.format_exc())
        (RUN/'edit_responses'/p.name).write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
