"""Answer extraction copied from the pinned author protocol mathqa.py (pure helpers only).
No dataset loader, benchmark content, or published result is imported.
"""

def answer_text(text):
    """The text the extraction reads: a stray reasoning block cut off (up to the last '</think>')."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text

def _box_at(text, idx):
    """the content of the box that starts at text[idx] ('\\boxed' or '\\fbox'), or None."""
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):   # brace matched, as last_boxed_only_string
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    out = r[1:k].strip()
                    return out or None
        return None          # not closed (e.g. a response cut inside the box)
    if key == "\\boxed" and rest.startswith(" "):   # '\\boxed 5': up to the next '$' or line end (Qwen's reading)
        out = rest.strip().split("$")[0].split("\n")[0].strip()
        return out or None
    return None

def last_boxed(text):
    """The content of the last box of a text, or None: hendrycks last_boxed_only_string + remove_boxed (the last
    '\\boxed', else the last '\\fbox'; brace matched; an unclosed last box = None), extended to '\\fbox{..}',
    '\\boxed {..}' and '\\boxed 5' (no braces: read up to the next '$' or line end, as Qwen's evaluation reads it).
    An empty box = None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)

def extract_answer(text):
    return last_boxed(answer_text(text))
