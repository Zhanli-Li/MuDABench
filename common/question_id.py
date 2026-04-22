import argparse
import hashlib
import json
from typing import Any, Dict, List


def _normalize_for_hash(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize_for_hash(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, list):
        return [_normalize_for_hash(item) for item in value]
    return value


def build_question_id(question: Any, metadata: List[Dict[str, Any]]) -> str:
    payload = {
        "question": str(question or "").strip(),
        "metadata": _normalize_for_hash(metadata or []),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "q_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def resolve_question_id(item: Dict[str, Any]) -> str:
    explicit = item.get("question_id")
    if explicit not in (None, ""):
        return str(explicit)
    return build_question_id(item.get("question"), item.get("metadata") or [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stable question_id values from question + metadata")
    parser.add_argument("--dataset", required=True, help="path to dataset json list")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise ValueError("dataset must be a json list")

    rows = []
    for idx, item in enumerate(dataset):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "index": idx,
                "question_id": resolve_question_id(item),
                "question": item.get("question"),
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
