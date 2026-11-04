from pathlib import Path
import json

turn = Path("results/astra_reproduction_20261104/raw/turns/turn_029/answer.md")
text = turn.read_text()
start = text.index("```python") + len("```python")
end = text.index("```", start)
code = text[start:end].lstrip("\n")
request = {
    "code": code,
}
out = Path("results/astra_reproduction_20261104/raw/editor_work/edit_requests/0020_final_search_state.json")
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
out.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
print(out)
