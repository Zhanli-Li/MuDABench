import json
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

BASE_URL = os.getenv("CHATDOC_BASE_URL", "https://api.chatdoc.studio")
API_KEY = os.getenv("CHATDOC_API_KEY", "")
CACHE_PATH = os.getenv(
    "CHATDOC_APP_CACHE_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chatdoc_app_cache.json"),
)

_APP_CACHE_LOCK = threading.Lock()
_APP_CACHE: Dict[str, str] = {}
_APP_KEY_LOCKS: Dict[str, threading.Lock] = {}


def _load_cache_from_disk() -> None:
    if not os.path.exists(CACHE_PATH):
        return
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, str) and v:
                    _APP_CACHE[k] = v
    except Exception:
        return


def _persist_cache_to_disk() -> None:
    cache_dir = os.path.dirname(CACHE_PATH)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    tmp = f"{CACHE_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_APP_CACHE, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CACHE_PATH)


with _APP_CACHE_LOCK:
    _load_cache_from_disk()


def _request_json(
    method: str,
    url: str,
    api_key: str,
    json_payload: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if params:
        q = urllib.parse.urlencode(params)
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{q}"

    body = json.dumps(json_payload).encode("utf-8") if json_payload is not None else None
    headers = {"Authorization": f"Bearer {api_key}"}
    if json_payload is not None:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url=url, data=body, method=method, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            status = resp.getcode()
    except Exception as e:
        raise RuntimeError(f"HTTP request failed: {method} {url}; err={e}")

    try:
        data = json.loads(raw)
    except Exception:
        data = {"raw": raw}

    if status >= 400:
        raise RuntimeError(f"HTTP {status} {method} {url}; response={data}")
    return data if isinstance(data, dict) else {"data": data}


def _extract_success_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if data.get("code") and data.get("code") != "success":
        raise RuntimeError(f"chatdoc returned non-success: {data}")
    payload = data.get("data")
    return payload if isinstance(payload, dict) else {}


def _create_chat_app_only(doc_id: str, base_url: str, api_key: str) -> str:
    url = f"{base_url}/v1/chat/apps"
    app_name = f"kbqa-{doc_id}"[:30]
    payload_candidates: List[Dict[str, Any]] = [
        {
            "name": app_name,
            "instruction": "Answer questions strictly based on the provided knowledge source.",
            "use_case": "knowledge_base_qa",
            "source_traceable": True,
            "support_new_conversation": True,
            "sources": [{"id": doc_id}],
        },
        {
            "name": app_name,
            "instruction": "Answer questions strictly based on the provided knowledge source.",
            "use_case": "knowledge_base_qa",
            "source_traceable": True,
            "support_new_conversation": True,
            "sources": {"id": doc_id},
        },
    ]

    errors: List[str] = []
    for payload in payload_candidates:
        try:
            resp = _request_json("POST", url, api_key=api_key, json_payload=payload, timeout=60)
            app_data = _extract_success_data(resp)
            app_id = str(app_data.get("id") or "").strip()
            if app_id:
                return app_id
            raise RuntimeError(f"create chat app missing id: {resp}")
        except Exception as e:
            errors.append(f"payload={payload} -> {e}")

    raise RuntimeError(
        f"create_app failed for doc_id={doc_id}. tried {len(payload_candidates)} payload(s). errors={errors}"
    )


def _publish_chat_app(app_id: str, base_url: str, api_key: str) -> None:
    url = f"{base_url}/v1/chat/apps/{app_id}/publish"
    resp = _request_json("POST", url, api_key=api_key, timeout=60)
    if resp.get("code") and resp.get("code") != "success":
        raise RuntimeError(f"publish chat app failed: {resp}")


def create_app(doc_id: str, base_url: str, api_key: str) -> str:
    app_id = _create_chat_app_only(doc_id=doc_id, base_url=base_url, api_key=api_key)
    _publish_chat_app(app_id=app_id, base_url=base_url, api_key=api_key)
    return app_id


def _cache_key(doc_id: str, base_url: str, api_key: str) -> str:
    return f"{base_url}|chat|{doc_id}|{api_key[:16]}"


def _get_cache_lock(key: str) -> threading.Lock:
    with _APP_CACHE_LOCK:
        lk = _APP_KEY_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _APP_KEY_LOCKS[key] = lk
        return lk


def _get_or_create_cached_app_id(doc_id: str, base_url: str, api_key: str) -> str:
    key = _cache_key(doc_id, base_url, api_key)
    lk = _get_cache_lock(key)
    with lk:
        with _APP_CACHE_LOCK:
            cached = _APP_CACHE.get(key)
        if cached:
            return cached

        app_id = create_app(doc_id=doc_id, base_url=base_url, api_key=api_key)
        with _APP_CACHE_LOCK:
            _APP_CACHE[key] = app_id
            try:
                _persist_cache_to_disk()
            except Exception:
                pass
        return app_id


def _invalidate_cached_app_id(
    doc_id: str,
    base_url: str,
    api_key: str,
    app_id: Optional[str] = None,
) -> None:
    key = _cache_key(doc_id, base_url, api_key)
    with _APP_CACHE_LOCK:
        if key not in _APP_CACHE:
            return
        if app_id is None or _APP_CACHE.get(key) == app_id:
            _APP_CACHE.pop(key, None)
            try:
                _persist_cache_to_disk()
            except Exception:
                pass


def _should_recreate_app(err: Exception) -> bool:
    s = str(err).lower()
    return (
        ("404" in s and "/v1/chat/apps/" in s)
        or ("not found" in s)
        or (("app_id" in s or "app id" in s) and "invalid" in s)
    )


def _create_conversation(app_id: str, base_url: str, api_key: str) -> str:
    url = f"{base_url}/v1/chat/apps/{app_id}/conversations"
    resp = _request_json("POST", url, api_key=api_key, timeout=60)
    conv_data = _extract_success_data(resp)
    conversation_id = str(conv_data.get("id") or "").strip()
    if not conversation_id:
        raise RuntimeError(f"create_conversation missing id: {resp}")
    return conversation_id


def _send_message(
    app_id: str,
    conversation_id: str,
    question: str,
    base_url: str,
    api_key: str,
) -> str:
    url = f"{base_url}/v1/chat/apps/{app_id}/messages"
    payload = {
        "conversation_id": conversation_id,
        "question": question,
    }
    resp = _request_json(
        "POST",
        url,
        api_key=api_key,
        json_payload=payload,
        timeout=120,
        params={"stream": "false"},
    )
    msg_data = _extract_success_data(resp)
    answer = msg_data.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    if isinstance(msg_data.get("content"), str) and msg_data["content"].strip():
        return msg_data["content"]
    raise RuntimeError(f"send_message missing answer: {resp}")


def ask_chatdoc(
    document_id: str,
    prompt: str,
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    max_retries: int = 3,
) -> str:
    if not api_key:
        raise RuntimeError("CHATDOC_API_KEY is empty. Set CHATDOC_API_KEY in env first.")

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        app_id: Optional[str] = None
        try:
            app_id = _get_or_create_cached_app_id(document_id, base_url=base_url, api_key=api_key)
            conversation_id = _create_conversation(app_id=app_id, base_url=base_url, api_key=api_key)
            return _send_message(
                app_id=app_id,
                conversation_id=conversation_id,
                question=prompt,
                base_url=base_url,
                api_key=api_key,
            )
        except Exception as e:
            last_err = e
            if app_id and _should_recreate_app(e):
                _invalidate_cached_app_id(document_id, base_url=base_url, api_key=api_key, app_id=app_id)
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))

    raise RuntimeError(f"ask_chatdoc failed after {max_retries} attempts: {last_err}")
