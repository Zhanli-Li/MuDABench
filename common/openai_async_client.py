"""
Compatibility wrapper:
- Prefer official `openai.AsyncOpenAI`
- Fallback to direct HTTP implementation when openai SDK is unavailable

This keeps CLI reproducible in environments where pip/openai install is blocked.
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


def _namespace_from_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _namespace_from_dict(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_namespace_from_dict(x) for x in d]
    return d


try:
    from openai import AsyncOpenAI as _OfficialAsyncOpenAI  # type: ignore

    AsyncOpenAI = _OfficialAsyncOpenAI
    OPENAI_SDK_AVAILABLE = True
except Exception:
    OPENAI_SDK_AVAILABLE = False

    import subprocess
    import urllib.request

    class _ChatCompletions:
        def __init__(self, parent: "AsyncOpenAI"):
            self._parent = parent

        async def create(
            self,
            model: str,
            messages: List[Dict[str, Any]],
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            response_format: Optional[Dict[str, Any]] = None,
        ) -> Any:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if temperature is not None:
                payload["temperature"] = temperature
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if response_format is not None:
                payload["response_format"] = response_format

            body = json.dumps(payload).encode("utf-8")
            url = self._parent.base_url.rstrip("/") + "/chat/completions"
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self._parent.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            def _do_request() -> Dict[str, Any]:
                with urllib.request.urlopen(req, timeout=self._parent.timeout_sec) as resp:
                    raw = resp.read().decode("utf-8")
                    return json.loads(raw)

            import asyncio

            try:
                data = await asyncio.to_thread(_do_request)
            except Exception:
                # Fallback: use curl to avoid Python HTTPS stack issues in some environments.
                cmd = [
                    "curl",
                    "-sS",
                    "-X",
                    "POST",
                    url,
                    "-H",
                    f"Authorization: Bearer {self._parent.api_key}",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    json.dumps(payload, ensure_ascii=False),
                ]
                proc = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self._parent.timeout_sec,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"curl failed: {proc.stderr.strip()}")
                data = json.loads(proc.stdout)
                if isinstance(data, dict) and "error" in data:
                    err = data.get("error")
                    if isinstance(err, dict):
                        raise RuntimeError(f"api error: {err.get('message', err)}")
                    raise RuntimeError(f"api error: {err}")
            return _namespace_from_dict(data)

    class _Chat:
        def __init__(self, parent: "AsyncOpenAI"):
            self.completions = _ChatCompletions(parent)

    class AsyncOpenAI:  # fallback class
        def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout_sec: float = 120.0):
            self.base_url = base_url or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") or ""
            self.timeout_sec = timeout_sec
            self.chat = _Chat(self)
