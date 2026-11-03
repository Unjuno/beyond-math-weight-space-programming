"""Offline, separately served forward-only approximation of Experiment 8.

No training, no benchmark loading, no access to the published edit. Requests and
responses are persisted. Successful tool calls, not individual model operations,
consume the 50-pass budget (as in the original protocol).
"""
import argparse, collections, hashlib, json, os, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/astra_clean_20261103'
BASE = ROOT / 'model-cache/Qwen3.5-0.8B'
PRACTICE = Path('<AUTHOR_LOGS_DIR>/exp8_math/practice')
INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'

def append(path, obj):
    with open(path, 'a') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush(); os.fsync(f.fileno())

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def practice():
    rows = {}
    for name in ['gsm8k_train', 'math_train']:
        with open(PRACTICE / (name + '.jsonl')) as f:
            for line in f:
                row = json.loads(line); rows[row['id']] = row
    return rows

def init():
    for name in ['requests','responses','edited','frozen','edit_requests','edit_responses']:
        (RUN/name).mkdir(parents=True, exist_ok=True)
    if (RUN/'manifest.json').exists(): return
    rows = practice()
    # Entire sources are partitioned by id hash before their problem text is read.
    groups = collections.defaultdict(list)
    for ident, r in rows.items():
        source = r['source']
        groups[source].append(ident)
    splits = {k:[] for k in ['diagnostic','validation','confirmation10','confirmation25','confirmation50']}
    for source, ids in sorted(groups.items()):
        ids.sort(key=lambda s:hashlib.sha256(('astra-clean-v1|' + s).encode()).digest())
        offset = 0
        for name,count in [('diagnostic',32),('validation',32),('confirmation10',64),('confirmation25',64),('confirmation50',64)]:
            splits[name] += ids[offset:offset+count]; offset += count
    manifest = {'created_at':time.time(),'base':str(BASE),'base_sha256':digest(BASE/'model.safetensors-00001-of-00001.safetensors'),
        'practice_sha256':{n:digest(PRACTICE/(n+'.jsonl')) for n in ['gsm8k_train','math_train']},
        'splits':splits,'split_seed':'astra-clean-v1','max_passes':50,
        'sampling':{'enable_thinking':True,'temperature':0.6,'top_p':0.95,'top_k':20,'presence_penalty':1.5},
        'deviations':['MPS BF16 Transformers replaces unavailable original forward backend',
        'deterministic SHA256 item/sample seeds reconstructed; original seed algorithm unavailable',
        'local tool implementations and software isolation replace original persistent Linux sandbox',
        'batch sizes and token caps chosen for four-hour wall-clock budget; not full-scale benchmark replication']}
    (RUN/'manifest.json').write_text(json.dumps(manifest,indent=2))
    append(RUN/'rationale.jsonl',{'time':time.time(),'phase':'init','hypothesis':None,
        'note':'New experiment; no published edit/result inspected; all diagnostic and confirmation IDs are TRAIN-only. Confirmation text is not exposed until frozen candidate is evaluated.'})

def last_boxed(text):
    if '</think>' in text: text=text.rsplit('</think>',1)[1]
    ix=text.rfind('\\boxed')
    if ix<0: ix=text.rfind('\\fbox')
    if ix<0:return None
    rest=text[ix:].split('{',1)
    if len(rest)<2:return None
    depth=1;out=''
    for ch in rest[1]:
        if ch=='{':depth+=1
        elif ch=='}':depth-=1
        if depth==0:return out.strip()
        out+=ch
    return None

def grade(pred,gold):
    if pred is None:return False
    try:
        from math_verify import parse,verify
        return bool(verify(parse('\\boxed{'+str(gold)+'}'),parse('\\boxed{'+pred+'}')))
    except ImportError:
        # Explicitly provisional; regrade with the author's verifier when present.
        def norm(s):return str(s).replace(' ','').replace(',','').replace('\\left','').replace('\\right','').replace('\\dfrac','\\frac').strip('$')
        return norm(pred)==norm(gold)

def worker():
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration,LogitsProcessor,StoppingCriteria
    torch.set_grad_enabled(False);torch.set_num_threads(6)
    init(); rows=practice()
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('mps')
    params=dict(model.named_parameters()); active=set()
    original=safe_open(str(BASE/'model.safetensors-00001-of-00001.safetensors'),framework='pt',device='cpu')
    layers=model.model.language_model.layers
    class Presence(LogitsProcessor):
        def __init__(self,start):self.start=start
        def __call__(self,input_ids,scores):
            ids=input_ids[0,self.start:]
            if ids.numel(): scores[0,torch.unique(ids)]-=1.5
            return scores
    class Progress(StoppingCriteria):
        def __init__(self,start,ident,variant):self.start=start;self.ident=ident;self.variant=variant;self.time=time.time()
        def __call__(self,input_ids,scores,**kwargs):
            n=input_ids.shape[1]-self.start
            if n%256==0:
                print(json.dumps({'progress_id':self.ident,'model':self.variant,'tokens':n,'seconds':time.time()-self.time}),flush=True)
            return False
    def apply(which,edit_dir):
        nonlocal active
        paths=list(Path(edit_dir).glob('*.npy')) if which=='edited' else []
        now={p.stem for p in paths}
        for name in active|now:
            tensor=torch.from_numpy(np.load(Path(edit_dir)/(name+'.npy'))) if name in now else original.get_tensor(name)
            params[name].copy_(tensor.to(device='mps',dtype=params[name].dtype))
        active=now
    class Capture:
        def __init__(self,indices,teacher=False):
            self.teacher=teacher;self.data={};self.handles=[]
            for idx in indices:
                self.data[idx]={'windows':[],'sums':None,'count':0,'prompt':None}
                self.handles.append(layers[idx].mlp.down_proj.register_forward_pre_hook(self.hook(idx)))
        def hook(self,idx):
            def f(module,args):
                x=args[0][0].detach().float();d=self.data[idx]
                if self.teacher:d['windows'].append(x.cpu().numpy());return
                if x.shape[0]>1:d['prompt']=x[-1].cpu().numpy();return
                x=x[0]
                d['sums']=x.clone() if d['sums'] is None else d['sums']+x
                d['count']+=1
                if d['count']==128:
                    d['windows'].append((d['sums']/d['count']).cpu().numpy());d['sums']=None;d['count']=0
            return f
        def save(self,path):
            arrays={}
            for idx,d in self.data.items():
                if d['count']:d['windows'].append((d['sums']/d['count']).cpu().numpy())
                arrays[f'layer_{idx}']=np.concatenate(d['windows'],axis=0) if self.teacher and d['windows'] else np.stack(d['windows']) if d['windows'] else np.empty((0,3584),np.float32)
                if d['prompt'] is not None:arrays[f'prompt_{idx}']=d['prompt']
            np.savez_compressed(path,**arrays)
            for h in self.handles:h.remove()
    (RUN/'worker_ready.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time(),'device':'mps','dtype':'bfloat16'}))
    print('READY',flush=True)
    while True:
        pending=[p for p in sorted((RUN/'requests').glob('*.json')) if not (RUN/'responses'/p.name).exists()]
        if not pending:time.sleep(.5);continue
        path=pending[0];req=json.loads(path.read_text());start=time.time()
        ledger=list(open(RUN/'pass_ledger.jsonl')) if (RUN/'pass_ledger.jsonl').exists() else []
        used=sum(json.loads(x)['status']=='success' for x in ledger)
        if used>=50:raise RuntimeError('50-pass budget exhausted')
        k=used+1;outdir=RUN/f'pass_{k:02d}';outdir.mkdir(exist_ok=True)
        response={'pass':k,'request_file':str(path),'status':'running','start':start}
        append(RUN/'audit_events.jsonl',{**response,'request':req})
        try:
            assert req['type'] in ['generate','teacher_forced']
            edit_dir=req.get('edit_dir',str(RUN/'edited'))
            edit_hash={p.stem:digest(p) for p in Path(edit_dir).glob('*.npy')}
            response['edit_sha256']=edit_hash
            if req['type']=='generate':
                ids=req['problems'];n=req.get('n',1);variants=['original','edited'] if req['model']=='both' else [req['model']]
                assert len(ids)*n*len(variants)<=512
                cap=req.get('max_tokens',32768);assert 0<cap<=32768
                all_rows={}
                for variant in variants:
                    apply(variant,edit_dir);records=[]
                    target=outdir/(variant+'.jsonl')
                    if target.exists():raise RuntimeError('Refuse overwrite of existing generation evidence')
                    for ident in ids:
                        assert ident in rows
                        r=rows[ident];prompt=tok.apply_chat_template([{'role':'user','content':r['problem']+'\n'+INSTRUCTION}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
                        inputs=tok.apply_chat_template([{'role':'user','content':r['problem']+'\n'+INSTRUCTION}],tokenize=True,return_dict=True,return_tensors='pt',add_generation_prompt=True,enable_thinking=True).to('mps');plen=inputs.input_ids.shape[1]
                        for sample in range(n):
                            seed=int.from_bytes(hashlib.sha256(f'astra-v1|{ident}|{sample}'.encode()).digest()[:4],'little')
                            torch.manual_seed(seed);torch.mps.manual_seed(seed)
                            capture=Capture(req.get('mlp_layers',[]))
                            t0=time.time()
                            with torch.inference_mode():
                                output=model.generate(**inputs,max_new_tokens=cap,do_sample=True,temperature=.6,top_p=.95,top_k=20,
                                    logits_processor=[Presence(plen)],stopping_criteria=[Progress(plen,ident,variant)],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
                            generated=output[0,plen:].cpu().tolist();text=tok.decode(generated,skip_special_tokens=False)
                            pred=last_boxed(text);correct=grade(pred,r['answer'])
                            cutoff=len(generated)>=cap and generated[-1]!=tok.eos_token_id
                            seq=[tuple(generated[i:i+8]) for i in range(max(0,len(generated)-7))]
                            repeated=1-len(set(seq))/max(1,len(seq))
                            rec={'id':ident,'source':r['source'],'sample':sample,'seed':seed,'model':variant,'prompt':prompt,'prompt_token_ids':inputs.input_ids[0].cpu().tolist(),
                                'text':text,'token_ids':generated,'prediction':pred,'gold':r['answer'],'correct':correct,'tokens':len(generated),
                                'cutoff':cutoff,'failure':'correct' if correct else 'cutoff' if cutoff else 'no_box' if pred is None else 'wrong_answer',
                                'repeat_8gram_fraction':repeated,'seconds':time.time()-t0}
                            capture.save(outdir/f'{variant}_{ident}_{sample}_mlp.npz');append(target,rec);records.append(rec)
                            print(json.dumps({k:rec[k] for k in ['id','model','tokens','correct','failure','seconds']}),flush=True)
                    all_rows[variant]=records
                response['summary']={v:{'n':len(rs),'correct':sum(r['correct'] for r in rs),'accuracy':np.mean([r['correct'] for r in rs]),
                    'mean_tokens':np.mean([r['tokens'] for r in rs]),'cutoff':sum(r['cutoff'] for r in rs),'failure_counts':dict(collections.Counter(r['failure'] for r in rs))} for v,rs in all_rows.items()}
                if len(variants)==2:
                    dif=np.array([int(b['correct'])-int(a['correct']) for a,b in zip(all_rows['original'],all_rows['edited'])])
                    # Aggregate n samples within an item before standard error.
                    item=dif.reshape(len(ids),n).mean(1)
                    response['paired']={'delta':float(item.mean()),'se':float(item.std(ddof=1)/np.sqrt(len(item))) if len(item)>1 else None,
                        'wrong_to_right':int((dif==1).sum()),'right_to_wrong':int((dif==-1).sum()),'n_items':len(ids)}
            else:
                assert req['model'] in ['original','edited'];apply(req['model'],edit_dir)
                texts=req['texts'];assert 0<len(texts)<=1024;total=0;metrics=[]
                for ix,item in enumerate(texts):
                    text=item['text'] if isinstance(item,dict) else item
                    inp=tok(text,return_tensors='pt').to('mps');length=inp.input_ids.shape[1];total+=length
                    assert length<=32768 and total<=524288
                    capture=Capture(req.get('mlp_layers',[]),teacher=True)
                    with torch.inference_mode():res=model(**inp,use_cache=False)
                    lp=torch.log_softmax(res.logits[0,:-1].float(),dim=-1).gather(1,inp.input_ids[0,1:,None]).squeeze(1).cpu().numpy()
                    capture.save(outdir/f'text_{ix:04d}_mlp.npz')
                    np.savez_compressed(outdir/f'text_{ix:04d}_scores.npz',logprobs=lp,token_ids=inp.input_ids.cpu().numpy()[0])
                    metric={'index':ix,'label':item.get('label',str(ix)) if isinstance(item,dict) else str(ix),'tokens':length,'mean_logprob':float(lp.mean())}
                    metrics.append(metric);del res;torch.mps.empty_cache()
                response['summary']={'texts':len(texts),'tokens':total,'metrics':metrics}
            response.update(status='success',seconds=time.time()-start,output_dir=str(outdir))
        except Exception:
            response.update(status='failed',seconds=time.time()-start,error=traceback.format_exc())
            print(response['error'],flush=True)
        append(RUN/'pass_ledger.jsonl',response)
        (RUN/'responses'/path.name).write_text(json.dumps(response,indent=2))
        print(json.dumps(response),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['init','worker']);a=p.parse_args()
    init() if a.mode=='init' else worker()
