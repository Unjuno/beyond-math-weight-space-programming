#!/usr/bin/env python3
"""Summarize paired task-level results from run_nonmath_probe.py."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from run_nonmath_probe import final_answer, score

def boot_delta(a,b,seed=20261103,nboot=20000):
    a=np.asarray(a,float); b=np.asarray(b,float)
    if len(a)!=len(b): raise ValueError('unpaired arrays')
    rng=np.random.default_rng(seed); draws=np.empty(nboot)
    for start in range(0,nboot,500):
        n=min(500,nboot-start); idx=rng.integers(0,len(a),(n,len(a)))
        draws[start:start+n]=(b[idx]-a[idx]).mean(axis=1)
    return float(np.mean(b-a)), tuple(float(x) for x in np.quantile(draws,[.025,.975]))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('results/raw/nonmath_probe.jsonl')); p.add_argument('--out',type=Path,default=Path('results/nonmath_probe_summary.csv')); p.add_argument('--mode',choices=['thinking','no_thinking'],default='no_thinking'); p.add_argument('--expected-items',type=int,default=24); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    task_map={json.loads(x)['id']:json.loads(x) for x in Path('configs/nonmath_probe.jsonl').read_text().splitlines() if x.strip()}
    # Re-score only completed final answers; raw logs stay immutable.
    for row in rows:
        answer=final_answer(row['output'],thinking=row['thinking'])
        checks=score(row['output'],task_map[row['id']]['checks'],thinking=row['thinking'])
        row['final_answer_reached']=bool(answer)
        row['all_checks_pass']=bool(answer) and bool(checks) and all(checks.values())
        row['checks']=checks
    by={(x['id'],x['condition']):x for x in rows}
    if len(by)!=len(rows): raise ValueError('duplicate (task, condition) rows')
    tasks=sorted({x['id'] for x in rows}); suffix='_thinking' if a.mode=='thinking' else '_no_thinking'; conditions=['base'+suffix,'edited'+suffix]
    if len(tasks)!=a.expected_items or any((t,c) not in by for t in tasks for c in conditions):
        raise ValueError(f'incomplete paired {a.mode} run: {len(tasks)} tasks, {len(rows)} rows (expected {a.expected_items})')
    metrics={
        'all_checks_pass':lambda r:float(r['all_checks_pass']),
        'cap_hit':lambda r:float(r['cap_hit']),
        'ended_with_eos':lambda r:float(r['ended_with_eos']),
        'final_answer_reached':lambda r:float(r['final_answer_reached']),
        'repeat_4gram':lambda r:float(r['repeat_4gram']),
        'generated_tokens':lambda r:float(r['generated_tokens']),
    }
    comparisons=[(conditions[0],conditions[1])]
    out=[]
    for fam in ['all']+sorted({by[(t,conditions[0])]['family'] for t in tasks}):
        ids=[t for t in tasks if fam=='all' or by[(t,conditions[0])]['family']==fam]
        for c in conditions:
            subset=[by[(t,c)] for t in ids]
            out.append({'scope':fam,'metric':'all_checks_pass','condition':c,'n':len(ids),'mean':np.mean([x['all_checks_pass'] for x in subset]),'comparison':'','delta':None,'paired_bootstrap_95_low':None,'paired_bootstrap_95_high':None,'cap_rate':np.mean([x['cap_hit'] for x in subset]),'eos_rate':np.mean([x['ended_with_eos'] for x in subset]),'repeat_rate':np.mean([x['repeat_4gram'] for x in subset]),'mean_tokens':np.mean([x['generated_tokens'] for x in subset])})
    for left,right in comparisons:
        for metric,get in metrics.items():
            x=[get(by[(t,left)]) for t in tasks]; y=[get(by[(t,right)]) for t in tasks]
            delta,(lo,hi)=boot_delta(x,y,seed=20261103+len(out))
            out.append({'scope':'all','metric':metric,'condition':right,'n':len(tasks),'mean':np.mean(y),'comparison':f'{right} - {left}','delta':delta,'paired_bootstrap_95_low':lo,'paired_bootstrap_95_high':hi,'cap_rate':None,'eos_rate':None,'repeat_rate':None,'mean_tokens':None})
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]),lineterminator='\n'); w.writeheader(); w.writerows(out)
    print('Task-family pass rates:')
    for r in out:
        if r['metric']=='all_checks_pass' and not r['comparison']:
            print(f"{r['scope']:24s} {r['condition']:20s} pass={r['mean']:.3f} cap={r['cap_rate']:.3f} eos={r['eos_rate']:.3f} repeat={r['repeat_rate']:.3f} tokens={r['mean_tokens']:.1f}")
    print('Paired overall comparisons:')
    for r in out:
        if r['comparison'] and r['metric']=='all_checks_pass': print(f"{r['comparison']:42s} pass Δ={r['delta']:+.3f} 95%=[{r['paired_bootstrap_95_low']:+.3f},{r['paired_bootstrap_95_high']:+.3f}]")
    print(f'wrote {a.out}')

if __name__=='__main__': main()
