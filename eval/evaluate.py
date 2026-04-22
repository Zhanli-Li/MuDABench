import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.openai_async_client import AsyncOpenAI

from prompts.eval_prompts import FINAL_PROMPT, INFO_PROMPT, ROW_METRIC_ONLY_PROMPT
from common.json_utils import robust_parse_json
from common.question_id import resolve_question_id


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-v3.2")
judge_client = AsyncOpenAI(
    base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

SOURCE_TABLE_HEADER_FIELDS = [
    "row_index",
    "entity_id",
    "year",
    "fact_type",
    "values",
    "details",
    "raw_text",
]

SCORABLE_DETAIL_EXCLUDE_KEYS = {
    "aligned_from_span",
    "span_role",
    "source_span_years",
    "source_span_fact_type",
    "paired_year",
    "year_source",
}


def _json_compact(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        return str(obj)


def _escape_md_cell(text: Any) -> str:
    s = str(text).replace("\n", " ").strip()
    return s.replace("|", "\\|")


def format_source_table_for_judge(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(empty source_table)"
    header = "| row_index | entity_id | year | fact_type | values | details | raw_text |"
    sep = "|---:|---|---:|---|---|---|---|"
    lines = [header, sep]
    for row in rows:
        year = row.get("year")
        if year is None and row.get("start_year") is not None and row.get("end_year") is not None:
            year = f"{row.get('start_year')}-{row.get('end_year')}"
        lines.append(
            "| {row_index} | {entity_id} | {year} | {fact_type} | {values} | {details} | {raw_text} |".format(
                row_index=_escape_md_cell(row.get("row_index")),
                entity_id=_escape_md_cell(row.get("entity_id")),
                year=_escape_md_cell(year if year is not None else ""),
                fact_type=_escape_md_cell(row.get("fact_type")),
                values=_escape_md_cell(_json_compact(row.get("values", {}))),
                details=_escape_md_cell(_json_compact(row.get("details", {}))),
                raw_text=_escape_md_cell(row.get("raw_text", "")),
            )
        )
    return "\n".join(lines)


def format_source_headers_for_judge() -> str:
    return _json_compact(SOURCE_TABLE_HEADER_FIELDS)


def _safe_year_int(v: Any) -> Optional[int]:
    try:
        if v is None or v == "":
            return None
        return int(str(v))
    except Exception:
        return None


def _normalize_year_value(v: Any) -> Any:
    yr = _safe_year_int(v)
    return yr if yr is not None else v


def _guess_doc_id(meta: Dict[str, Any]) -> Optional[str]:
    for k in ("chatdoc_upload_id", "doc_id", "document_id", "toBid", "id"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


def _pick_atomic_fact_keys(meta: Dict[str, Any]) -> List[str]:
    schema = meta.get("schema")
    keys: List[str] = []
    if isinstance(schema, dict):
        keys = [str(k) for k in schema if isinstance(k, str) and (k.startswith("value_") or k.startswith("detail_"))]
    if not keys:
        keys = [str(k) for k in meta.keys() if isinstance(k, str) and (k.startswith("value_") or k.startswith("detail_"))]
    return sorted(set(keys))


def _derive_fact_type_from_values(values: Dict[str, Any]) -> str:
    metric_name = values.get("metric_name")
    if metric_name not in (None, ""):
        return str(metric_name)
    value_keys = sorted(str(k) for k in values.keys())
    if not value_keys:
        return "atomic_metadata"
    return value_keys[0]


def is_atomic_metadata_entry(meta: Dict[str, Any]) -> bool:
    return bool(_pick_atomic_fact_keys(meta or {}))


def is_atomic_metadata_item(item: Dict[str, Any]) -> bool:
    metadata = item.get("metadata") or []
    return bool(metadata) and any(isinstance(meta, dict) and is_atomic_metadata_entry(meta) for meta in metadata)


def metadata_entry_to_source_row(metadata_entry: Dict[str, Any], row_index: int) -> Dict[str, Any]:
    fact_keys = _pick_atomic_fact_keys(metadata_entry)
    values: Dict[str, Any] = {}
    details: Dict[str, Any] = {}
    for key in fact_keys:
        if key not in metadata_entry:
            continue
        if key.startswith("value_"):
            values[key[len("value_") :]] = metadata_entry.get(key)
        elif key.startswith("detail_"):
            details[key[len("detail_") :]] = metadata_entry.get(key)

    symbol = metadata_entry.get("symbol")
    year = _normalize_year_value(metadata_entry.get("year"))
    fact_type = _derive_fact_type_from_values(values)
    raw_text = (
        "Synthetic row reconstructed from atomic metadata: "
        f"symbol={symbol}, year={year}, fact_type={fact_type}, values={_json_compact(values)}, details={_json_compact(details)}"
    )
    return {
        "row_index": row_index,
        "entity_id": symbol,
        "year": year,
        "fact_type": fact_type,
        "values": values,
        "details": details,
        "raw_text": raw_text,
        "row_source": "atomic_metadata",
        "doc_id": _guess_doc_id(metadata_entry),
    }


def build_eval_rows(item: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    if is_atomic_metadata_item(item):
        metadata = item.get("metadata") or []
        rows = [
            metadata_entry_to_source_row(meta, row_index=idx)
            for idx, meta in enumerate(metadata)
            if isinstance(meta, dict)
        ]
        return rows, "atomic_metadata"
    return [], "legacy"


def _pick_best_attempt(attempts: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not attempts:
        return None
    successful = [a for a in attempts if a.get("success") and a.get("answer")]
    pool = successful or [a for a in attempts if a.get("answer")] or attempts
    pool = sorted(pool, key=lambda x: (str(x.get("ts", "")), int(x.get("attempt", 0))))
    return pool[-1] if pool else None


def _stringify_doc_info(value: Any) -> str:
    if isinstance(value, str):
        return value
    return _json_compact(value)


def _normalize_doc_interaction_list(items: List[Any]) -> List[Dict[str, Any]]:
    qa_items: List[Dict[str, Any]] = []
    for idx, item in enumerate(items or []):
        if isinstance(item, dict):
            prompt = item.get("prompt") or item.get("question") or item.get("query") or item.get("field") or "Extracted information"
            answer = item.get("answer")
            if answer is None:
                answer = item.get("result")
            if answer is None:
                answer = item.get("content")
            if answer is None:
                answer = item.get("text")
            if answer is None:
                answer = item
            ts = str(item.get("ts", item.get("timestamp", "")))
            attempt = int(item.get("attempt", idx) or 0)
        else:
            prompt = "Extracted information"
            answer = item
            ts = ""
            attempt = idx
        answer_text = _stringify_doc_info(answer).strip()
        if not answer_text:
            continue
        qa_items.append(
            {
                "ts": ts,
                "attempt": attempt,
                "prompt": str(prompt),
                "answer": answer_text,
            }
        )
    return qa_items


def _normalize_legacy_plan_bucket(plan_bucket: Dict[str, Any]) -> List[Dict[str, Any]]:
    qa_items: List[Dict[str, Any]] = []
    for _, record in (plan_bucket or {}).items():
        if not isinstance(record, dict):
            continue
        best = _pick_best_attempt(record.get("attempts") or [])
        if not best or not best.get("prompt") or not best.get("answer"):
            continue
        qa_items.append(
            {
                "ts": str(best.get("ts", "")),
                "attempt": int(best.get("attempt", 0) or 0),
                "prompt": best.get("prompt"),
                "answer": best.get("answer"),
            }
        )
    return qa_items


def load_doc_conversations_from_run_log(
    agent_log_dir: Optional[str],
    metadata: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    if not agent_log_dir:
        return {}
    run_log_path = os.path.join(agent_log_dir, "run_log.json")
    if not os.path.exists(run_log_path):
        return {}

    try:
        with open(run_log_path, "r", encoding="utf-8") as f:
            run_log = json.load(f)
    except Exception:
        return {}

    metadata_by_doc_id: Dict[str, Dict[str, Any]] = {}
    for meta in metadata or []:
        doc_id = _guess_doc_id(meta)
        if doc_id:
            metadata_by_doc_id[str(doc_id)] = meta

    doc_interactions = run_log.get("doc_interactions") or {}
    doc_records: Dict[str, Dict[str, Any]] = {}
    for doc_id, doc_payload in doc_interactions.items():
        if isinstance(doc_payload, list):
            qa_items = _normalize_doc_interaction_list(doc_payload)
        elif isinstance(doc_payload, dict):
            qa_items = _normalize_legacy_plan_bucket(doc_payload)
        else:
            qa_items = _normalize_doc_interaction_list([doc_payload])
        qa_items.sort(key=lambda x: (x["ts"], x["attempt"], x["prompt"]))
        conversation = "\n".join(f"Q: {item['prompt']}\nA: {item['answer']}" for item in qa_items).strip()
        meta = metadata_by_doc_id.get(str(doc_id), {})
        doc_records[str(doc_id)] = {
            "doc_id": str(doc_id),
            "meta": meta,
            "conversation": conversation,
            "qa_items": qa_items,
        }

    for doc_id, meta in metadata_by_doc_id.items():
        doc_records.setdefault(
            str(doc_id),
            {"doc_id": str(doc_id), "meta": meta, "conversation": "", "qa_items": []},
        )
    return doc_records


def resolve_row_to_doc_id(
    row: Dict[str, Any],
    metadata: List[Dict[str, Any]],
    doc_conversations: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    row_doc_id = row.get("doc_id") or _guess_doc_id(row)
    if row_doc_id is not None:
        row_doc_id = str(row_doc_id)
        for meta in metadata or []:
            doc_id = _guess_doc_id(meta)
            if doc_id and str(doc_id) == row_doc_id:
                return row_doc_id, meta
        if row_doc_id in doc_conversations:
            return row_doc_id, doc_conversations.get(row_doc_id, {}).get("meta")

    row_entity = str(row.get("entity_id", "")).strip()
    row_year = _safe_year_int(row.get("year"))
    candidates: List[Tuple[int, int, str, Dict[str, Any]]] = []

    for idx, meta in enumerate(metadata or []):
        symbol = str(meta.get("symbol", "")).strip()
        if row_entity and symbol != row_entity:
            continue
        meta_year = _safe_year_int(meta.get("year"))
        score = 0
        if row_entity and symbol == row_entity:
            score += 10
        if row_year is not None and meta_year == row_year:
            score += 5
        if row_year is None:
            score += 1
        doc_id = _guess_doc_id(meta)
        if doc_id and doc_conversations.get(str(doc_id), {}).get("conversation"):
            score += 2
        if doc_id:
            candidates.append((score, -idx, str(doc_id), meta))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    _, _, doc_id, meta = candidates[0]
    return doc_id, meta


def build_row_metadata_columns(doc_meta: Optional[Dict[str, Any]], row: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta = doc_meta or {}
    cols: List[Dict[str, Any]] = []
    fields = [
        ("symbol", meta.get("symbol") if meta.get("symbol") is not None else row.get("entity_id")),
        ("year", meta.get("year") if meta.get("year") is not None else row.get("year")),
        ("doctype", meta.get("doctype")),
    ]
    for field, value in fields:
        if value in (None, ""):
            continue
        cols.append({"field": field, "expected_value": value})
    return cols


def build_row_metric_columns(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    cols: List[Dict[str, Any]] = []
    for key in sorted((row.get("values") or {}).keys()):
        cols.append({"field": f"values.{key}", "expected_value": row["values"][key]})
    for key in sorted((row.get("details") or {}).keys()):
        if key in SCORABLE_DETAIL_EXCLUDE_KEYS:
            continue
        cols.append({"field": f"details.{key}", "expected_value": row["details"][key]})
    if not cols:
        cols.append({"field": "fact_type", "expected_value": row.get("fact_type")})
    return cols


def format_columns_for_judge(columns: List[Dict[str, Any]], value_label: str = "expected_value") -> str:
    if not columns:
        return "(empty columns)"
    header = f"| field | {value_label} |"
    sep = "|---|---|"
    lines = [header, sep]
    for col in columns:
        lines.append(
            "| {field} | {expected_value} |".format(
                field=_escape_md_cell(col.get("field")),
                expected_value=_escape_md_cell(_json_compact(col.get("expected_value"))),
            )
        )
    return "\n".join(lines)


def _format_single_row_for_judge(row: Dict[str, Any]) -> str:
    return format_source_table_for_judge([row])


def _compact_doc_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {}
    keep = ("symbol", "year", "doctype", "chatdoc_upload_id", "toBid", "id")
    return {k: meta.get(k) for k in keep if meta.get(k) is not None}


def _build_log_key(question: Any, metadata: List[Dict[str, Any]]) -> str:
    q = str(question or "").strip()
    compact_meta: List[Dict[str, Any]] = []
    for meta in metadata or []:
        if not isinstance(meta, dict):
            continue
        compact_meta.append(_compact_doc_meta(meta))
    meta_fp = _json_compact(compact_meta)
    return f"{q} || metadata={meta_fp}"


def _output_question_id(out: Dict[str, Any]) -> Optional[str]:
    for key in ("question_id", "qid"):
        value = out.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def build_output_index(agent_outputs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for out in agent_outputs or []:
        if not isinstance(out, dict):
            continue
        qid = _output_question_id(out)
        if qid:
            indexed[qid] = out
    return indexed


def validate_agent_outputs(agent_outputs: List[Dict[str, Any]], eval_mode: str) -> None:
    seen_qids = set()
    for idx, out in enumerate(agent_outputs or []):
        if not isinstance(out, dict):
            raise ValueError(f"agent-output[{idx}] must be a json object")
        qid = _output_question_id(out)
        if not qid:
            raise ValueError(f"agent-output[{idx}] must include question_id or qid")
        if qid in seen_qids:
            raise ValueError(f"duplicate question_id/qid in agent-output: {qid}")
        seen_qids.add(qid)
        if eval_mode == "rag":
            extra_keys = set(out.keys()) - {"question_id", "qid", "retrieved_chunks", "model_answer"}
            if extra_keys:
                raise ValueError(
                    f"rag agent-output[{idx}] has unsupported keys: {sorted(extra_keys)}; "
                    "allowed keys are question_id, qid, retrieved_chunks, model_answer"
                )
            chunks = out.get("retrieved_chunks")
            if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks):
                raise ValueError(f"rag agent-output[{idx}].retrieved_chunks must be a list of strings")
            if not isinstance(out.get("model_answer"), str):
                raise ValueError(f"rag agent-output[{idx}].model_answer must be a string")


async def call_json_judge(prompt: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    resp = await judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful Judge."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = robust_parse_json(raw, expected_type="dict")
    return raw, parsed


async def run_judge_with_rescue(
    prompt: str,
    default_obj: Dict[str, Any],
    max_retries: int = 4,
    timeout_sec: Optional[float] = None,
) -> Dict[str, Any]:
    if timeout_sec is None:
        try:
            timeout_sec = float(os.getenv("JUDGE_TIMEOUT", "40"))
        except Exception:
            timeout_sec = 40.0

    errors: List[str] = []
    last_raw = ""
    for attempt in range(max_retries + 1):
        raw = ""
        parsed = None
        try:
            raw, parsed = await asyncio.wait_for(call_json_judge(prompt), timeout=timeout_sec)
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")
        last_raw = raw or last_raw

        if isinstance(parsed, dict):
            return {
                "raw": raw,
                "parsed": parsed,
                "meta": {
                    "attempts": attempt + 1,
                    "errors": errors,
                    "fallback_used": False,
                },
            }
        if attempt < max_retries:
            await asyncio.sleep(0.5 * (2 ** attempt))

    return {
        "raw": last_raw,
        "parsed": default_obj,
        "meta": {
            "attempts": max_retries + 1,
            "errors": errors,
            "fallback_used": True,
        },
    }


def normalize_row_judgment(
    parsed: Dict[str, Any],
    row_index: int,
    metadata_columns: List[Dict[str, Any]],
    metric_columns: List[Dict[str, Any]],
    metadata_aligned: bool = True,
) -> Dict[str, Any]:
    def _to_str_list(v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            try:
                obj = json.loads(v)
                if isinstance(obj, list):
                    return [str(x).strip() for x in obj if str(x).strip()]
            except Exception:
                pass
            return [x.strip() for x in re.split(r"[,;\n]+", v) if x.strip()]
        return []

    metadata_fields = [str(col.get("field")) for col in metadata_columns if col.get("field")]
    metric_fields = [str(col.get("field")) for col in metric_columns if col.get("field")]

    correct_metadata_fields = metadata_fields[:] if metadata_aligned else []
    identity_match = bool(metadata_aligned)

    correct_metric_fields = sorted(set(f for f in _to_str_list(parsed.get("correct_metric_fields")) if f in metric_fields))
    if not identity_match:
        correct_metric_fields = []

    metric_total = max(1, len(metric_fields))
    metric_correct_count = len(correct_metric_fields)

    row_correct = identity_match and metric_correct_count == metric_total

    return {
        "row_index": row_index,
        "metadata_judge_mode": "implicit_aligned" if metadata_aligned else "alignment_missing",
        "identity_match": identity_match,
        "metadata_correct_count": len(correct_metadata_fields),
        "metadata_total": len(metadata_fields),
        "metric_correct_count": metric_correct_count,
        "metric_total": metric_total,
        "correct_metric_fields": correct_metric_fields,
        "row_correct": row_correct,
    }


def normalize_info_dict(parsed: Dict[str, Any], total_required: int) -> Dict[str, Any]:
    def _to_int(v: Any, default: int = 0) -> int:
        try:
            if isinstance(v, bool):
                return int(v)
            return int(float(v))
        except Exception:
            return default

    ce = _to_int(parsed.get("correct_extractions", parsed.get("correct", 0)), 0)
    tr = _to_int(parsed.get("total_required", parsed.get("total", total_required or 1)), total_required or 1)
    tr = tr or 1
    if ce > tr:
        ce = tr
    return {
        "correct_extractions": ce,
        "total_required": tr,
        "explanation": parsed.get("explanation", ""),
    }


def enrich_info_judgment(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed or {})

    def _to_int(v: Any, default: int = 0) -> int:
        try:
            if isinstance(v, bool):
                return int(v)
            if v is None or v == "":
                return default
            return int(float(v))
        except Exception:
            return default

    row_correct_count = _to_int(out.get("row_correct_count", out.get("correct_extractions", 0)))
    row_total = _to_int(out.get("row_total", out.get("total_required", 0)))
    column_correct_count = _to_int(out.get("column_correct_count", 0))
    column_total = _to_int(out.get("column_total", 0))
    cell_correct_count = _to_int(out.get("cell_correct_count", out.get("metric_correct_count", 0)))
    cell_total = _to_int(out.get("cell_total", out.get("metric_total", 0)))
    metadata_correct_count = _to_int(out.get("metadata_correct_count", 0))
    metadata_total = _to_int(out.get("metadata_total", 0))

    out.update(
        {
            "correct_extractions": row_correct_count,
            "total_required": row_total,
            "row_correct_count": row_correct_count,
            "row_total": row_total,
            "row_accuracy": (row_correct_count / row_total) if row_total else 0.0,
            "column_correct_count": column_correct_count,
            "column_total": column_total,
            "column_accuracy": (column_correct_count / column_total) if column_total else 0.0,
            "cell_correct_count": cell_correct_count,
            "cell_total": cell_total,
            "cell_accuracy": (cell_correct_count / cell_total) if cell_total else 0.0,
            "metric_correct_count": cell_correct_count,
            "metric_total": cell_total,
            "metric_accuracy": (cell_correct_count / cell_total) if cell_total else 0.0,
            "metadata_correct_count": metadata_correct_count,
            "metadata_total": metadata_total,
            "metadata_accuracy": (metadata_correct_count / metadata_total) if metadata_total else 0.0,
        }
    )
    return out


def normalize_final_dict(parsed: Dict[str, Any]) -> Dict[str, Any]:
    val = parsed.get("is_correct")
    if isinstance(val, str):
        is_corr = val.strip().lower().rstrip(".") in {"true", "yes", "y", "1", "correct", "right", "是", "对", "正确"}
    elif isinstance(val, (int, float, bool)):
        is_corr = bool(val)
    else:
        is_corr = False
    return {
        "is_correct": is_corr,
        "explanation": parsed.get("explanation", ""),
    }


async def judge_source_table_document_wise(
    question: str,
    source_table: List[Dict[str, Any]],
    metadata: List[Dict[str, Any]],
    agent_log_dir: Optional[str],
    judge_concurrency: int = 8,
) -> Dict[str, Any]:
    doc_conversations = load_doc_conversations_from_run_log(agent_log_dir, metadata)
    grouped_rows: Dict[str, Dict[str, Any]] = {}

    for row in source_table or []:
        doc_id, matched_meta = resolve_row_to_doc_id(row, metadata, doc_conversations)
        if doc_id is None:
            doc_id = f"unmatched::{row.get('row_index')}"
            matched_meta = None

        group = grouped_rows.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "dataset_meta": matched_meta or {},
                "doc_meta": matched_meta or doc_conversations.get(doc_id, {}).get("meta") or {},
                "rows": [],
                "conversation": doc_conversations.get(doc_id, {}).get("conversation", ""),
            },
        )
        group["rows"].append(row)

    row_jobs: List[Dict[str, Any]] = []
    judge_concurrency = max(1, int(judge_concurrency or 1))
    judge_semaphore = asyncio.Semaphore(judge_concurrency)

    async def _run_row_judge(prompt: str) -> Dict[str, Any]:
        async with judge_semaphore:
            return await run_judge_with_rescue(prompt, {"correct_metric_fields": []})

    judge_tasks = []
    for group in grouped_rows.values():
        rows = sorted(group["rows"], key=lambda x: int(x.get("row_index", 0)))
        for row in rows:
            metadata_columns = build_row_metadata_columns(group.get("dataset_meta"), row)
            metric_columns = build_row_metric_columns(row)
            aligned_doc_meta = _compact_doc_meta(group.get("dataset_meta") or group.get("doc_meta"))
            prompt = ROW_METRIC_ONLY_PROMPT.format(
                question=question,
                source_headers=format_source_headers_for_judge(),
                row_index=int(row.get("row_index", 0)),
                source_row=_format_single_row_for_judge(row),
                aligned_doc_meta=_json_compact(aligned_doc_meta) if aligned_doc_meta else "{}",
                metric_total=len(metric_columns),
                metric_columns=format_columns_for_judge(metric_columns),
                agent_conversation=group.get("conversation") or "(empty conversation)",
            )
            row_jobs.append(
                {
                    "doc_id": group.get("doc_id"),
                    "row": row,
                    "prompt": prompt,
                    "metadata_columns": metadata_columns,
                    "metric_columns": metric_columns,
                    "alignment_found": not str(group.get("doc_id", "")).startswith("unmatched::"),
                    "conversation_turns": len(doc_conversations.get(group.get("doc_id"), {}).get("qa_items", [])),
                }
            )
            judge_tasks.append(_run_row_judge(prompt))

    judge_results = await asyncio.gather(*judge_tasks) if judge_tasks else []

    doc_judgments: List[Dict[str, Any]] = []
    matched_union: List[int] = []
    missing_union: List[int] = []
    total_correct_rows = 0
    total_required_rows = 0
    total_correct_metadata = 0
    total_required_metadata = 0
    total_correct_metrics = 0
    total_required_metrics = 0
    column_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"correct_count": 0, "total": 0, "row_indices": []})

    for job, res in zip(row_jobs, judge_results):
        row = job["row"]
        row_index = int(row.get("row_index", 0))
        parsed_norm = normalize_row_judgment(
            res.get("parsed") or {},
            row_index=row_index,
            metadata_columns=job["metadata_columns"],
            metric_columns=job["metric_columns"],
            metadata_aligned=job["alignment_found"],
        )

        total_required_rows += 1
        total_correct_rows += int(parsed_norm.get("row_correct", False))
        total_correct_metadata += int(parsed_norm.get("metadata_correct_count", 0))
        total_required_metadata += int(parsed_norm.get("metadata_total", 0))
        total_correct_metrics += int(parsed_norm.get("metric_correct_count", 0))
        total_required_metrics += int(parsed_norm.get("metric_total", 0))

        correct_metric_field_set = set(parsed_norm.get("correct_metric_fields", []))
        for col in job.get("metric_columns", []):
            field = str(col.get("field", "")).strip()
            if not field:
                continue
            column_stats[field]["total"] += 1
            column_stats[field]["row_indices"].append(row_index)
            if field in correct_metric_field_set:
                column_stats[field]["correct_count"] += 1

        if parsed_norm.get("row_correct"):
            matched_union.append(row_index)
        else:
            missing_union.append(row_index)

        doc_judgments.append(
            {
                "doc_id": job.get("doc_id"),
                "row_index": row_index,
                "fact_type": row.get("fact_type"),
                "alignment_found": job.get("alignment_found"),
                "conversation_turns": job.get("conversation_turns", 0),
                "prompt": job.get("prompt"),
                "raw_response": res.get("raw", ""),
                "parsed": parsed_norm,
                "meta": res.get("meta", {}),
            }
        )

    matched_set = sorted(set(int(x) for x in matched_union))
    all_row_indices = sorted(int(row.get("row_index")) for row in source_table or [] if row.get("row_index") is not None)
    missing_set = sorted(set(int(x) for x in missing_union)) or [x for x in all_row_indices if x not in matched_set]

    column_records = []
    for field in sorted(column_stats.keys()):
        stat = column_stats[field]
        total = int(stat.get("total", 0))
        correct = int(stat.get("correct_count", 0))
        column_records.append(
            {
                "field": field,
                "correct_count": correct,
                "total": total,
                "accuracy": (correct / total) if total else 0.0,
                "row_indices": sorted(set(int(x) for x in stat.get("row_indices", []))),
                "column_correct": bool(total and correct == total),
            }
        )

    column_correct_count = sum(1 for item in column_records if item.get("column_correct"))
    column_total = len(column_records)
    aggregate = {
        "correct_extractions": total_correct_rows,
        "total_required": total_required_rows or max(1, len(source_table or [])),
        "matched_row_indices": matched_set,
        "missing_row_indices": missing_set,
        "metadata_mode": "implicit_aligned",
        "metadata_correct_count": total_correct_metadata,
        "metadata_total": total_required_metadata,
        "metadata_accuracy": (total_correct_metadata / total_required_metadata) if total_required_metadata else 0.0,
        "row_correct_count": total_correct_rows,
        "row_total": total_required_rows or max(1, len(source_table or [])),
        "row_accuracy": (total_correct_rows / total_required_rows) if total_required_rows else 0.0,
        "column_correct_count": column_correct_count,
        "column_total": column_total,
        "column_accuracy": (column_correct_count / column_total) if column_total else 0.0,
        "cell_correct_count": total_correct_metrics,
        "cell_total": total_required_metrics,
        "cell_accuracy": (total_correct_metrics / total_required_metrics) if total_required_metrics else 0.0,
        "metric_correct_count": total_correct_metrics,
        "metric_total": total_required_metrics,
        "metric_accuracy": (total_correct_metrics / total_required_metrics) if total_required_metrics else 0.0,
        "judge_mode": "document_row_wise_metric_only",
        "explanation": f"document-wise metric-only judge over {len(grouped_rows)} documents with implicit metadata alignment",
    }

    return {"aggregate": aggregate, "doc_judgments": doc_judgments, "doc_count": len(grouped_rows)}


async def process_dataset(
    dataset: List[Dict[str, Any]],
    agent_outputs: List[Dict[str, Any]],
    eval_mode: str = "agent",
    judge_concurrency: int = 8,
) -> Dict[str, Any]:
    log_data: Dict[str, Any] = {}
    info_judgments: List[Dict[str, Any]] = []
    final_judgments: List[Dict[str, Any]] = []
    output_by_qid = build_output_index(agent_outputs)

    for i, item in enumerate(dataset):
        question_id = resolve_question_id(item)
        out = output_by_qid.get(question_id)
        if out is None:
            raise ValueError(f"missing agent-output record for question_id={question_id}")
        question = item.get("question")
        metadata = item.get("metadata") or []
        log_key = _build_log_key(question, metadata)
        log_data[log_key] = {
            "question_id": question_id,
            "question": question,
            "metadata_fingerprint": _json_compact([_compact_doc_meta(meta) for meta in metadata if isinstance(meta, dict)]),
            "agent": out,
            "reference": {
                "source_answer": item.get("source_answer"),
                "final_answer": item.get("final_answer"),
            },
        }

        eval_rows, eval_source = build_eval_rows(item)
        if eval_mode == "agent" and eval_rows:
            doc_wise = await judge_source_table_document_wise(
                question=question,
                source_table=eval_rows,
                metadata=metadata,
                agent_log_dir=out.get("agent_log_dir") or out.get("log_dir"),
                judge_concurrency=judge_concurrency,
            )
            parsed_norm = enrich_info_judgment(doc_wise["aggregate"])
            info_judgments.append(parsed_norm)
            log_data[log_key]["judge_info"] = {
                "mode": doc_wise["aggregate"].get("judge_mode", "document_row_wise_metric_only"),
                "parsed": parsed_norm,
                "meta": {"doc_count": doc_wise.get("doc_count", 0), "eval_source": eval_source, "row_count": len(eval_rows)},
            }
            log_data[log_key]["judge_info_rows"] = doc_wise.get("doc_judgments", [])
        else:
            if eval_mode == "rag":
                rag_chunks = out.get("retrieved_chunks") or []
                if rag_chunks and not isinstance(rag_chunks, list):
                    raise ValueError(f"retrieved_chunks must be a list for question_id={question_id}")
                rag_text = "\n".join(str(x) for x in rag_chunks)
                agent_conversation = f"RAG retrieved chunks:\n{rag_text}"
            else:
                agent_conversation = (
                    f"JSON data:{out.get('json_data', '')}\n"
                    f"JSON data schema:{out.get('json_dec', '')}\n"
                    f"Analysis code:{out.get('code', '')}\n"
                    f"Analysis code response:{out.get('code_resp', '')}\n"
                )

            prompt = INFO_PROMPT.format(
                len_source=len(item.get("source_answer", [])),
                source_answer=item.get("source_answer"),
                agent_conversation=agent_conversation,
            )
            total_required = max(1, len(item.get("source_answer", [])))
            res = await run_judge_with_rescue(prompt, {"correct_extractions": 0, "total_required": total_required})
            parsed_norm = enrich_info_judgment(normalize_info_dict(res.get("parsed") or {}, total_required))
            info_judgments.append(parsed_norm)
            log_data[log_key]["judge_info"] = {
                "mode": "legacy_question_wise" if eval_mode == "agent" else "rag_info_prompt",
                "prompt": prompt,
                "raw_response": res.get("raw", ""),
                "parsed": parsed_norm,
                "meta": res.get("meta", {}),
            }

        fp = FINAL_PROMPT.format(
            question=question,
            final_answer=item.get("final_answer"),
            chatdoc_answer=out.get("model_answer", out.get("result", "None")),
        )
        fres = await run_judge_with_rescue(fp, {"is_correct": False})
        fparsed = normalize_final_dict(fres.get("parsed") or {})
        final_judgments.append(fparsed)
        log_data[log_key]["judge_final"] = {
            "prompt": fp,
            "raw_response": fres.get("raw", ""),
            "parsed": fparsed,
            "meta": fres.get("meta", {}),
        }

    row_correct_count = sum(int(j.get("row_correct_count", 0)) for j in info_judgments)
    row_total = sum(int(j.get("row_total", 0)) for j in info_judgments)
    row_accuracy = row_correct_count / row_total if row_total else 0.0

    column_correct_count = sum(int(j.get("column_correct_count", 0)) for j in info_judgments)
    column_total = sum(int(j.get("column_total", 0)) for j in info_judgments)
    column_accuracy = column_correct_count / column_total if column_total else 0.0

    cell_correct_count = sum(int(j.get("cell_correct_count", 0)) for j in info_judgments)
    cell_total = sum(int(j.get("cell_total", 0)) for j in info_judgments)
    cell_accuracy = cell_correct_count / cell_total if cell_total else 0.0

    final_correct = sum(1 for j in final_judgments if j.get("is_correct", False))
    final_total = len(final_judgments)
    final_accuracy = final_correct / final_total if final_total else 0.0

    solid_correct = 0
    for i in range(len(final_judgments)):
        final_ok = bool(final_judgments[i].get("is_correct", False))
        info_ok = bool(info_judgments[i].get("row_correct_count", 0) == info_judgments[i].get("row_total", 0))
        if final_ok and info_ok:
            solid_correct += 1

    solid_total = len(final_judgments)
    solid_accuracy = solid_correct / solid_total if solid_total else 0.0

    return {
        "log_data": log_data,
        "row_correct_count": row_correct_count,
        "row_total": row_total,
        "row_accuracy": row_accuracy,
        "column_correct_count": column_correct_count,
        "column_total": column_total,
        "column_accuracy": column_accuracy,
        "cell_correct_count": cell_correct_count,
        "cell_total": cell_total,
        "cell_accuracy": cell_accuracy,
        "final_correct_count": final_correct,
        "final_total": final_total,
        "final_accuracy": final_accuracy,
        "solid_correct_count": solid_correct,
        "solid_total": solid_total,
        "solid_accuracy": solid_accuracy,
    }


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


async def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MuDABench-style agent/rag outputs")
    parser.add_argument("--dataset", required=True, help="path to benchmark dataset json list")
    parser.add_argument("--agent-output", required=True, help="path to agent_runner output json list")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eval-mode", choices=["agent", "rag"], default="agent")
    parser.add_argument("--judge-concurrency", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all dataset items")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = _load_json(args.dataset)
    agent_output = _load_json(args.agent_output)

    if not isinstance(dataset, list) or not isinstance(agent_output, list):
        raise ValueError("dataset and agent-output must both be json list")
    validate_agent_outputs(agent_output, args.eval_mode)

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(dataset):
        import random

        rng = random.Random(args.seed)
        dataset = rng.sample(dataset, args.sample_size)

    results = await process_dataset(
        dataset,
        agent_output,
        eval_mode=args.eval_mode,
        judge_concurrency=args.judge_concurrency,
    )

    summary = {
        "row_accuracy": results["row_accuracy"],
        "row": f"{results['row_correct_count']}/{results['row_total']}",
        "column_accuracy": results["column_accuracy"],
        "column": f"{results['column_correct_count']}/{results['column_total']}",
        "cell_accuracy": results["cell_accuracy"],
        "cell": f"{results['cell_correct_count']}/{results['cell_total']}",
        "final_accuracy": results["final_accuracy"],
        "final": f"{results['final_correct_count']}/{results['final_total']}",
        "solid_accuracy": results["solid_accuracy"],
        "solid": f"{results['solid_correct_count']}/{results['solid_total']}",
    }

    _dump_json(os.path.join(args.output_dir, "eval_summary.json"), summary)
    _dump_json(os.path.join(args.output_dir, "eval_log.json"), results["log_data"])

    print(f"row accuracy: {summary['row_accuracy']:.4f} ({summary['row']})")
    print(f"column accuracy: {summary['column_accuracy']:.4f} ({summary['column']})")
    print(f"cell accuracy: {summary['cell_accuracy']:.4f} ({summary['cell']})")
    print(f"final accuracy: {summary['final_accuracy']:.4f} ({summary['final']})")
    print(f"solid accuracy: {summary['solid_accuracy']:.4f} ({summary['solid']})")
    print(f"saved: {os.path.join(args.output_dir, 'eval_summary.json')}")
    print(f"saved: {os.path.join(args.output_dir, 'eval_log.json')}")


if __name__ == "__main__":
    asyncio.run(cli_main())
