import re
from typing import Any, Dict

_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def safe_template_format(template: str, values: Dict[str, Any]) -> str:
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        return str(values[key]) if key in values else match.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs: Any) -> str:
        return safe_template_format(self.template, kwargs)
