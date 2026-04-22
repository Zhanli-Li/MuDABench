import re
import textwrap
from typing import Optional


def extract_python_code(llm_response: str) -> Optional[str]:
    if not llm_response or not isinstance(llm_response, str):
        return None

    text = llm_response.strip()
    extraction_patterns = [
        (r"<execute>(.*?)</execute>", re.DOTALL),
        (r"```python\s*\n(.*?)\n```", re.DOTALL),
        (r"```py\s*\n(.*?)\n```", re.DOTALL),
        (r"```\s*\n(.*?)\n```", re.DOTALL),
        (r"`([^`\n]+)`", 0),
    ]

    candidates = []
    for pattern, flags in extraction_patterns:
        matches = re.findall(pattern, text, flags)
        for m in matches:
            code = m[0] if isinstance(m, tuple) else m
            cleaned = clean_code(code)
            if is_valid_python_code(cleaned):
                candidates.append(cleaned)

    if not candidates:
        return None
    return max(candidates, key=len)


def clean_code(code: str) -> str:
    if not code:
        return ""
    code = textwrap.dedent(code.strip())
    for prefix in ["python", "py", "code:", "代码:", "执行:", "运行:"]:
        if code.lower().startswith(prefix.lower()):
            code = code[len(prefix) :].lstrip()
    return code


def is_valid_python_code(code: str) -> bool:
    if not code.strip():
        return False
    try:
        compile(code, "<string>", "exec")
        return True
    except Exception:
        return False
