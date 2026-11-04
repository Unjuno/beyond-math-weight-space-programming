#!/usr/bin/env python3
"""Score the frozen three-item p50c follow-up with the project's strict JSON rule."""
import hashlib, json
from collections import Counter
from collections import defaultdict
from pathlib import Path
from score_transfer_eval import strict_equal, parse_json_answer

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'results/raw/p50c_extraction_followup_20261104.jsonl'
TASKS=ROOT/'results/heldout24_p50c_tasks.json'
MANIFEST=ROOT/'results/raw/p50c_extraction_followup_20261104.manifest.json'
OUT=ROOT/'results/p50c_extraction_followup_20261104_scores.json'
ANSWERS=ROOT/'results/p50c_extraction_followup_20261104_answers.jsonl'

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def repeat8(ids):
 n=max(0,len(ids)-7)
 if not n:return 0.0
 counts=Counter(tuple(ids[i:i+8]) for i in range(n))
 return sum(v-1 for v in counts.values() if v>1)/n

tasks={t['id']:t for t in json.loads(TASKS.read_text())['tasks']}
pairs=defaultdict(dict)
for line in RAW.read_text().splitlines():
 r=json.loads(line); text=r['output']
 answer=text.rsplit('</think>',1)[1].strip() if '</think>' in text else ''
 parsed=parse_json_answer(answer)
 r['answer']=answer; r['strict_json_match']=parsed is not None and strict_equal(parsed,tasks[r['task_id']]['gold'])
 r['task_success']=bool(r['strict_json_match'] and r['ended_with_eos'] and not r['cap_hit'])
 r['repeat8_prefix1024']=repeat8(r['generated_token_ids'][:1024])
 pairs[(r['task_id'],r['replicate'])][r['arm']]=r
assert len(pairs)==9 and all(set(v)=={'base','edited'} for v in pairs.values())
rows=[]
for (task,rep),arms in sorted(pairs.items()):
 b,e=arms['base'],arms['edited']
 rows.append({'task_id':task,'replicate':rep,'seed':b['seed'],'base':{k:b[k] for k in ['generated_tokens','cap_hit','ended_with_eos','strict_json_match','task_success','repeat8_prefix1024']},'edited':{k:e[k] for k in ['generated_tokens','cap_hit','ended_with_eos','strict_json_match','task_success','repeat8_prefix1024']}})
summary={}
for task in ['x04','x05','x06']:
 rr=[r for r in rows if r['task_id']==task]
 summary[task]={arm:{'eos':sum(r[arm]['ended_with_eos'] and not r[arm]['cap_hit'] for r in rr),'strict_correct':sum(r[arm]['strict_json_match'] for r in rr),'task_success':sum(r[arm]['task_success'] for r in rr)} for arm in ['base','edited']}
summary['all_9_pairs']={arm:{k:sum(r[arm][k] for r in rows) for k in ['ended_with_eos','strict_json_match','task_success']} for arm in ['base','edited']}
delta8=[r['edited']['repeat8_prefix1024']-r['base']['repeat8_prefix1024'] for r in rows]
summary['paired_prefix1024_repeat8_delta']={'n':len(delta8),'mean_edited_minus_base':sum(delta8)/len(delta8),'negative_pairs':sum(x<0 for x in delta8),'positive_pairs':sum(x>0 for x in delta8),'ties':sum(x==0 for x in delta8)}
answer_rows=[]
for (task,rep),arms in sorted(pairs.items()):
 for arm,r in arms.items():
  answer_rows.append({'task_id':task,'seed':r['seed'],'arm':arm,'generated_tokens':r['generated_tokens'],'cap_hit':r['cap_hit'],'ended_with_eos':r['ended_with_eos'],'thinking_closed':r['thinking_closed'],'final_answer':r['answer']})
ANSWERS.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in answer_rows))
result={'protocol':'p50c-extraction-followup-v1','raw_sha256':sha(RAW),'task_file_sha256':sha(TASKS),'manifest_sha256':sha(MANIFEST),'sanitized_answers_sha256':sha(ANSWERS),'rows':rows,'summary':summary,'interpretation_rule':'The frozen absolute threshold (edited task_success >=2/3 per selected item) is descriptive; it does not require or establish improvement over base.'}
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
