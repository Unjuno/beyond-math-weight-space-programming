"""STATIC until pass 1 completes: one counted, fresh TRAIN prefix check.

Do not run concurrently with another forward request. A failed parity check is
still a successful inference call and consumes one pass. It does not authorize
batched scientific comparisons. No published edit or benchmark is accessible.
"""
import hashlib, json, os, time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
from astra_clean_forward import ROOT,RUN,BASE,INSTRUCTION,practice,append,last_boxed,grade


def main():
    import torch
    from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
    from astra_clean_batch_sampler_prototype import IndependentMpsSampler
    torch.set_grad_enabled(False);torch.set_num_threads(6)
    completed=json.loads((RUN/'responses/001_diagnostic.json').read_text())
    assert completed['status']=='success', 'Original full diagnostic must finish first'
    ledger=[json.loads(x) for x in open(RUN/'pass_ledger.jsonl')]
    k=1+sum(r['status']=='success' for r in ledger);assert k<=50
    req=json.loads((RUN/'requests/001_diagnostic.json').read_text())
    ids=req['problems'];assert len(ids)==4
    baseline={r['id']:r for r in map(json.loads,open(RUN/'pass_01/original.jsonl'))}
    assert set(ids)==set(baseline)
    outdir=RUN/f'pass_{k:02d}';outdir.mkdir(exist_ok=False)
    request={'type':'generate','model':'original','problems':ids,'n':1,'max_tokens':256,'backend':'independent_rng_batch_validation',
        'hypothesis':'Separately owned MPS RNG states may reproduce the fresh sequential token prefixes with less wall time. Reject batching on any prefix or prompt mismatch.'}
    (outdir/'request.json').write_text(json.dumps(request,indent=2))
    start=time.time();append(RUN/'audit_events.jsonl',{'start':start,'pass':k,'request':request})
    rows=practice();tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True);tok.padding_side='left'
    model=Qwen3_5ForConditionalGeneration.from_pretrained(BASE,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('mps')
    messages=[[{'role':'user','content':rows[ident]['problem']+'\n'+INSTRUCTION}] for ident in ids]
    inputs=tok.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt',padding=True,add_generation_prompt=True,enable_thinking=True).to('mps')
    seeds=[int.from_bytes(hashlib.sha256(f'astra-v1|{ident}|0'.encode()).digest()[:4],'little') for ident in ids]
    sampler=IndependentMpsSampler(seeds,inputs.input_ids.shape[1])
    t0=time.time()
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=256,do_sample=False,logits_processor=[sampler],eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,use_cache=True)
    torch.mps.synchronize();seconds=time.time()-t0;checks=[]
    for i,ident in enumerate(ids):
        generated=output[i,inputs.input_ids.shape[1]:].cpu().tolist()
        if tok.eos_token_id in generated:generated=generated[:generated.index(tok.eos_token_id)+1]
        prompt_ids=inputs.input_ids[i][inputs.attention_mask[i].bool()].cpu().tolist()
        gold_prefix=baseline[ident]['token_ids'][:len(generated)]
        equal=generated==gold_prefix
        first_difference=next((j for j,(a,b) in enumerate(zip(generated,gold_prefix)) if a!=b),None)
        check={'id':ident,'tokens':len(generated),'prefix_equal':equal,'first_difference':first_difference,
            'prompt_equal':prompt_ids==baseline[ident]['prompt_token_ids'],'seed_equal':seeds[i]==baseline[ident]['seed'],
            'eos_token_id':tok.eos_token_id,'cutoff':len(generated)==256 and generated[-1]!=tok.eos_token_id}
        checks.append(check)
        text=tok.decode(generated,skip_special_tokens=False);pred=last_boxed(text)
        append(outdir/'original.jsonl',{'id':ident,'source':rows[ident]['source'],'sample':0,'seed':seeds[i],
            'prompt_token_ids':prompt_ids,'token_ids':generated,'text':text,'prediction':pred,'gold':rows[ident]['answer'],
            'correct':grade(pred,rows[ident]['answer']),'tokens':len(generated),'cutoff':check['cutoff'],
            'note':'Implementation prefix check only. Truncated accuracy is not a mathematical-performance estimate.'})
    accepted=all(c['prefix_equal'] and c['prompt_equal'] and c['seed_equal'] for c in checks)
    result={'pass':k,'status':'success','start':start,'seconds':time.time()-start,'generation_seconds':seconds,'output_dir':str(outdir),
        'backend':'independent_rng_batch_validation','checks':checks,'prefix_validation_passed':accepted,
        'inference_limit':'256-token prefix agreement only; does not establish long-context numerical equivalence or benchmark improvement.'}
    append(RUN/'pass_ledger.jsonl',result);(outdir/'summary.json').write_text(json.dumps(result,indent=2))
    append(RUN/'rationale.jsonl',{'time':time.time(),'phase':'batch_prefix_validation','decision':'Eligible for reviewed batched follow-up' if accepted else 'Reject batch backend; keep sequential',**result})
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
