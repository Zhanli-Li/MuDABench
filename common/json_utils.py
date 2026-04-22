import ast
import json
import re
from typing import Any, Optional


def extract_json_from_code_fence(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    m = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return None


def find_brace_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def repair_json_like(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    t = re.sub(r"'([A-Za-z0-9_\- ]+)'\s*:", r'"\1":', t)
    t = re.sub(r":\s*'([^']*)'", r': "\1"', t)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    return t


def robust_parse_json(text: str, expected_type: Optional[str] = None) -> Optional[Any]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if expected_type == "list" and not isinstance(obj, list):
            return None
        if expected_type == "dict" and not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        pass

    fenced = extract_json_from_code_fence(text)
    if fenced:
        for cand in (fenced, repair_json_like(fenced)):
            try:
                obj = json.loads(cand)
                if expected_type == "list" and not isinstance(obj, list):
                    continue
                if expected_type == "dict" and not isinstance(obj, dict):
                    continue
                return obj
            except Exception:
                continue

    braced = find_brace_json(text)
    if braced:
        for cand in (braced, repair_json_like(braced)):
            try:
                obj = json.loads(cand)
                if expected_type == "list" and not isinstance(obj, list):
                    continue
                if expected_type == "dict" and not isinstance(obj, dict):
                    continue
                return obj
            except Exception:
                continue

    try:
        obj = ast.literal_eval(fenced or braced or text)
        if expected_type == "list" and not isinstance(obj, list):
            return None
        if expected_type == "dict" and not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    m = re.search(rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>", text)
    if not m:
        return None
    return m.group(1).strip()
