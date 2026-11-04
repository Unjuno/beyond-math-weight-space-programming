"""Rebuild a per-task completion ledger from committed evidence."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def repeat8(ids):
    n = 8
    total = max(0, len(ids) - n + 1)
    if total == 0:
        return 0.0
    counts = Counter(tuple(ids[i:i+n]) for i in range(total))
    return sum(count - 1 for count in counts.values() if count > 1) / total


def main():
    answers = [json.loads(line) for line in (ROOT / 'results/heldout24_p50c_final_answers.jsonl').read_text().splitlines()]
    raw = [json.loads(line) for line in (ROOT / 'results/raw/heldout24_p50c_fullcap.jsonl').read_text().splitlines()]
    pairs = {}
    for row in answers:
        arms = pairs.setdefault(row['id'], {})
        assert row['condition'] not in arms
        arms[row['condition']] = row
    repeats = {}
    for row in raw:
        repeats.setdefault(row['id'], {})[row['condition']] = repeat8(row['generated_token_ids'])
    assert len(pairs) == 24
    scores = json.loads((ROOT / 'results/heldout24_p50c_fullcap_scores.json').read_text())
    exact = {(r['task_id'], r['condition']): r['task_success'] for r in scores['structured_scores']}
    print('| Task | Base tokens | Edited tokens | EOS B→E | repeat8 B→E | Strict score B→E |')
    print('|---|---:|---:|---|---|---|')
    for task, arms in sorted(pairs.items()):
        assert set(arms) == {'base', 'edited'}
        b, e = arms['base'], arms['edited']
        assert b['seed'] == e['seed']
        grades = '/'.join(str(exact.get((task, arm), 'writing: separate ratings')) for arm in ('base', 'edited'))
        print(f"| {task} | {b['generated_tokens']} | {e['generated_tokens']} | {int(b['eos_before_cap'])}→{int(e['eos_before_cap'])} | {repeats[task]['base']:.3f}→{repeats[task]['edited']:.3f} | {grades} |")
    rescued = [k for k, v in pairs.items() if not v['base']['eos_before_cap'] and v['edited']['eos_before_cap']]
    print('\nCompletion transitions:', ', '.join(sorted(rescued)))
    print('These transitions alone do not establish loop suppression or correctness.')


if __name__ == '__main__':
    main()
