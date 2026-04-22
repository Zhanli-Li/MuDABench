import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.openai_async_client import AsyncOpenAI

from prompts.agent_prompts import (
    ANSWER_PROMPT,
    CODE_ACT_PROMPT,
    JSON_CONTINUE_PROMPT,
    JSON_EXTRACTION_PROMPT,
    PLAN_PROMPT,
)
from agent.extract_python_code import extract_python_code
from agent.extractors import resolve_extractor
from common.json_utils import extract_tag_content, robust_parse_json
from common.question_id import resolve_question_id


class LLMResponse:
    def __init__(self, text: str):
        self.text = text


class LLM(Protocol):
    async def acomplete(self, prompt: str) -> LLMResponse:
        ...


class OpenRouter:
    def __init__(self, model: str, max_tokens: int = 8192, temperature: float = 0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = AsyncOpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        )

    async def acomplete(self, prompt: str) -> LLMResponse:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        text = ""
        if resp and resp.choices:
            text = resp.choices[0].message.content or ""
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        return LLMResponse(str(text))


def _to_posix(p: str) -> str:
    return os.path.abspath(p).replace("\\", "/")


def _guess_doc_id(meta: Dict[str, Any]) -> Optional[str]:
    for k in ("chatdoc_upload_id", "doc_id", "document_id", "toBid", "id"):
        if k in meta and meta[k] is not None:
            return str(meta[k])
    return None


def _match_restriction_value(meta_value: Any, restriction_value: Any) -> bool:
    """
    Exact-match restriction check.
    - If restriction_value is a list/tuple/set, match if meta_value is exactly one item.
    - Otherwise match by exact string equality.
    """
    meta_s = str(meta_value)
    if isinstance(restriction_value, (list, tuple, set)):
        return any(meta_s == str(v) for v in restriction_value)
    return meta_s == str(restriction_value)


def _execute_python_code(code: str, timeout_sec: int = 120) -> str:
    if not code or not code.strip():
        return "no executable code found"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=os.getcwd(),
        )
        stdout = (proc.stdout or "").strip()
        stderr = (proc.stderr or "").strip()
        if proc.returncode == 0:
            return stdout if stdout else "(success with empty stdout)"
        return f"execution failed (code={proc.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    except subprocess.TimeoutExpired:
        return f"execution timeout > {timeout_sec}s"
    except Exception as e:
        return f"execution error: {e}"


def _redact(x: Any) -> Any:
    if isinstance(x, str):
        x = re.sub(r"sk-[a-zA-Z0-9_-]{8,}", lambda m: m.group(0)[:6] + "***" + m.group(0)[-3:], x)
        x = re.sub(r"ak--[a-zA-Z0-9_-]{8,}", lambda m: m.group(0)[:6] + "***" + m.group(0)[-3:], x)
        return x
    if isinstance(x, dict):
        return {k: _redact(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_redact(v) for v in x]
    return x


@dataclass
class DocQARecord:
    plan_id: str
    doc_id: str
    attempts: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RunLog:
    run_id: str
    qid: str
    task: str
    metadata_description: str
    started_at: str
    finished_at: Optional[str] = None
    status: str = "running"
    error: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    doc_interactions: Dict[str, Dict[str, DocQARecord]] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    def __init__(self, qid: str, task: str, metadata_description: str, base_dir: str):
        self.run_id = str(uuid.uuid4())
        self.qid = qid
        self.base_dir = _to_posix(os.path.join(base_dir, qid, self.run_id))
        os.makedirs(self.base_dir, exist_ok=True)
        self.log = RunLog(
            run_id=self.run_id,
            qid=qid,
            task=_redact(task),
            metadata_description=_redact(metadata_description),
            started_at=datetime.utcnow().isoformat(),
        )
        self._save()

    @property
    def path_json(self) -> str:
        return _to_posix(os.path.join(self.base_dir, "run_log.json"))

    def _save(self) -> None:
        payload = json.loads(json.dumps(self.log, default=lambda o: o.__dict__, ensure_ascii=False))
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def add_step(self, name: str, prompt: Any = None, response: Any = None, **extra: Any) -> None:
        self.log.steps.append(
            {
                "name": name,
                "ts": datetime.utcnow().isoformat(),
                "prompt": _redact(prompt),
                "response": _redact(response),
                "extra": _redact(extra),
            }
        )
        self._save()

    def add_doc_qa(self, doc_id: str, plan_id: str, prompt: str, answer: Optional[str], success: bool, attempt: int) -> None:
        bucket = self.log.doc_interactions.setdefault(doc_id, {})
        rec = bucket.get(plan_id)
        if rec is None:
            rec = DocQARecord(plan_id=plan_id, doc_id=doc_id)
            bucket[plan_id] = rec
        rec.attempts.append(
            {
                "ts": datetime.utcnow().isoformat(),
                "prompt": _redact(prompt),
                "answer": _redact(answer),
                "success": bool(success),
                "attempt": attempt,
                "extra": {},
            }
        )
        self._save()

    def add_artifact(self, name: str, content: Any) -> None:
        self.log.artifacts[name] = _redact(content)
        ext = os.path.splitext(name)[1]
        if not ext:
            ext = ".json" if isinstance(content, (dict, list)) else ".txt"
        p = os.path.join(self.base_dir, f"artifact_{os.path.splitext(name)[0]}{ext}")
        with open(p, "w", encoding="utf-8") as f:
            if isinstance(content, (dict, list)):
                json.dump(content, f, ensure_ascii=False, indent=2)
            else:
                f.write(str(content))
        self._save()

    def finish(self, status: str, error: Optional[str] = None) -> None:
        self.log.status = status
        self.log.error = _redact(error)
        self.log.finished_at = datetime.utcnow().isoformat()
        self._save()


class AsyncMutiDocAgent:
    def __init__(
        self,
        log_dir: str,
        extractor_backend: str = "chatdoc",
        extractor_entrypoint: Optional[str] = None,
        doc_concurrency: int = 1,
        plan_model: str = "deepseek/deepseek-r1-0528",
        norm_model: str = "deepseek/deepseek-chat-v3-0324",
        code_model: str = "deepseek/deepseek-r1-0528",
        answer_model: str = "deepseek/deepseek-r1-0528",
    ):
        self.log_dir = log_dir
        self.extractor = resolve_extractor(extractor_backend, extractor_entrypoint)
        self.plan_model = plan_model
        self.norm_model = norm_model
        self.code_model = code_model
        self.answer_model = answer_model
        self.doc_concurrency = max(1, int(doc_concurrency or 1))

    async def _call_llm(self, model: str, prompt: str) -> LLMResponse:
        llm = OpenRouter(model=model, max_tokens=60000, temperature=0.7)
        return await llm.acomplete(prompt)

    async def _call_llm_retry(self, model: str, prompt: str, logger: StructuredLogger, step_name: str) -> LLMResponse:
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                return await self._call_llm(model, prompt)
            except Exception as e:
                last_err = e
                logger.add_step(f"{step_name}_error", prompt=prompt, response=f"attempt={attempt}, err={e}")
                if attempt < 3:
                    await asyncio.sleep(2 ** (attempt - 1))
        raise RuntimeError(f"{step_name} failed: {last_err}")

    async def run_async(
        self,
        task: str,
        metadata_description: str,
        metadata: List[Dict[str, Any]],
        qid: Optional[str] = None,
    ) -> Dict[str, Any]:
        qid = qid or resolve_question_id({"question": task, "metadata": metadata})
        logger = StructuredLogger(qid=qid, task=task, metadata_description=metadata_description, base_dir=self.log_dir)
        doc_semaphore = asyncio.Semaphore(self.doc_concurrency)

        try:
            plan_prompt = PLAN_PROMPT.format(task=task, metadata_description=metadata_description)
            plan_resp = await self._call_llm_retry(self.plan_model, plan_prompt, logger, "plan")
            plan_json = robust_parse_json(plan_resp.text, expected_type="list")
            if not isinstance(plan_json, list) or not plan_json:
                plan_json = [{"subtask": "请提供关于{symbol}在{year}年{doctype}中的相关信息", "restriction": {}}]
            for p in plan_json:
                if isinstance(p, dict):
                    p.setdefault("plan_id", str(uuid.uuid4()))
            logger.add_step("plan", prompt=plan_prompt, response=plan_resp.text, plan_count=len(plan_json))

            doc_interactions: List[List[Dict[str, Any]]] = []
            for sub in plan_json:
                template = str(sub.get("subtask", ""))
                restriction = sub.get("restriction") or {}
                plan_id = str(sub.get("plan_id") or uuid.uuid4())

                one_plan_candidates: List[Dict[str, Any]] = []
                for meta in metadata:
                    ok = True
                    if isinstance(restriction, dict):
                        for rk, rv in restriction.items():
                            if rk in meta and not _match_restriction_value(meta[rk], rv):
                                ok = False
                                break
                    if not ok:
                        continue
                    doc_id = _guess_doc_id(meta)
                    if not doc_id:
                        continue
                    try:
                        prompt = template.format(**meta)
                    except Exception:
                        prompt = template
                    one_plan_candidates.append({"doc_id": doc_id, "prompt": prompt, "plan_id": plan_id})

                async def _ask_single_doc(doc_item: Dict[str, str]) -> Dict[str, Any]:
                    doc_id = doc_item["doc_id"]
                    prompt = doc_item["prompt"]
                    answer = None
                    success = False
                    for attempt in range(1, 4):
                        try:
                            async with doc_semaphore:
                                answer = await asyncio.to_thread(self.extractor, doc_id, prompt)
                            success = True
                            logger.add_doc_qa(
                                doc_id=doc_id,
                                plan_id=plan_id,
                                prompt=prompt,
                                answer=answer,
                                success=True,
                                attempt=attempt,
                            )
                            break
                        except Exception as e:
                            logger.add_doc_qa(
                                doc_id=doc_id,
                                plan_id=plan_id,
                                prompt=prompt,
                                answer=str(e),
                                success=False,
                                attempt=attempt,
                            )
                            if attempt < 3:
                                await asyncio.sleep(2 ** (attempt - 1))
                    return {
                        "doc_id": doc_id,
                        "plan_id": plan_id,
                        "prompt": prompt,
                        "answer": answer,
                        "success": success,
                    }

                if one_plan_candidates:
                    one_plan = await asyncio.gather(*(_ask_single_doc(item) for item in one_plan_candidates))
                else:
                    one_plan = []
                doc_interactions.append(one_plan)

            logger.add_step("branch", response={"groups": len(doc_interactions)})

            norm_results = []
            for one_plan in doc_interactions:
                conversations = [f"Q: {x['prompt']}\nA: {x['answer']}" for x in one_plan if x.get("answer")]
                if not conversations:
                    continue
                base_conv = "\n".join(conversations[:5])
                rest_conv = conversations[5:]

                norm_prompt = JSON_EXTRACTION_PROMPT.format(task=task, multi_conversation=base_conv)
                norm_resp = await self._call_llm_retry(self.norm_model, norm_prompt, logger, "norm_initial")
                logger.add_step("norm_initial", prompt=norm_prompt, response=norm_resp.text)

                json_content = extract_tag_content(norm_resp.text, "json")
                des_content = extract_tag_content(norm_resp.text, "des")
                parsed = robust_parse_json(json_content, expected_type="list") if json_content else None
                if parsed is None:
                    fallback = robust_parse_json(norm_resp.text, expected_type="dict")
                    if isinstance(fallback, dict) and isinstance(fallback.get("extracted_data"), list):
                        parsed = fallback.get("extracted_data")
                if not isinstance(parsed, list):
                    continue

                all_data = list(parsed)
                for conv in rest_conv:
                    cont_prompt = JSON_CONTINUE_PROMPT.format(json=json.dumps(parsed, ensure_ascii=False), new_conversation=conv)
                    cont_resp = await self._call_llm_retry(self.norm_model, cont_prompt, logger, "norm_continue")
                    logger.add_step("norm_continue", prompt=cont_prompt, response=cont_resp.text)
                    cont_json = extract_tag_content(cont_resp.text, "json")
                    cont_parsed = robust_parse_json(cont_json, expected_type="list") if cont_json else None
                    if isinstance(cont_parsed, list):
                        all_data.extend(cont_parsed)

                json_path = os.path.join(logger.base_dir, f"normed_{int(time.time())}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                logger.add_artifact("norm_json_path", json_path)
                norm_results.append(
                    {
                        "few_shot": parsed,
                        "all_data": all_data,
                        "json_path": _to_posix(json_path),
                        "schema": des_content or "",
                    }
                )

            if not norm_results:
                logger.add_step("code", response="no normalized data")
                logger.finish("failed", error="no normalized data")
                return {
                    "success": False,
                    "result": "",
                    "json_data": "",
                    "json_dec": "",
                    "code": "",
                    "code_resp": "no normalized data",
                    "qid": qid,
                    "run_id": logger.run_id,
                    "log_dir": logger.base_dir,
                }

            json_data_text = ""
            json_path_text = ""
            json_schema_text = ""
            full_data_text = ""
            for idx, item in enumerate(norm_results, start=1):
                json_data_text += f"No.{idx} json data:\n{json.dumps(item['few_shot'], ensure_ascii=False, indent=2)}\n"
                json_path_text += f"No.{idx} json path: {item['json_path']}\n"
                json_schema_text += f"No.{idx} schema: {item['schema'] or 'N/A'}\n"
                full_data_text += f"No.{idx} full data:\n{json.dumps(item['all_data'], ensure_ascii=False, indent=2)}\n"

            code_prompt = CODE_ACT_PROMPT.format(
                task=task,
                json_data=json_data_text,
                json_path=json_path_text,
                json_schema=json_schema_text,
            )
            code_resp_llm = await self._call_llm_retry(self.code_model, code_prompt, logger, "code_gen")
            logger.add_step("code_gen", prompt=code_prompt, response=code_resp_llm.text)
            code = extract_python_code(code_resp_llm.text) or ""
            code_exec = _execute_python_code(code)
            logger.add_artifact("analysis_code.py", code)
            logger.add_artifact("analysis_output.txt", code_exec)

            answer_prompt = ANSWER_PROMPT.format(task=task, data=json_schema_text, code=code, code_resp=code_exec)
            answer_resp = await self._call_llm_retry(self.answer_model, answer_prompt, logger, "final")
            logger.add_step("final", prompt=answer_prompt, response=answer_resp.text)
            logger.add_artifact("final_answer.txt", answer_resp.text)
            logger.finish("succeeded")

            return {
                "success": True,
                "result": answer_resp.text,
                "json_data": full_data_text,
                "json_dec": json_schema_text,
                "code": code,
                "code_resp": code_exec,
                "qid": qid,
                "run_id": logger.run_id,
                "log_dir": logger.base_dir,
            }

        except Exception as e:
            logger.add_step("run_error", response=str(e))
            logger.finish("failed", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "result": "",
                "json_data": "",
                "json_dec": "",
                "code": "",
                "code_resp": "",
                "qid": qid,
                "run_id": logger.run_id,
                "log_dir": logger.base_dir,
            }


async def run_single_query(agent: AsyncMutiDocAgent, item: Dict[str, Any], metadata_description: str) -> Dict[str, Any]:
    return await agent.run_async(
        task=item["question"],
        metadata_description=metadata_description,
        metadata=item["metadata"],
        qid=item.get("question_id") or item.get("qid"),
    )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


async def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run open-source reproducible multi-document agent")
    parser.add_argument("--dataset", required=True, help="path to dataset json list")
    parser.add_argument("--output", required=True, help="path to output jsonl")
    parser.add_argument("--metadata-description", default=(
        "Field 1: symbol, stock code. "
        "Field 2: year, year. "
        "Field 3: doctype, document type."
    ))
    parser.add_argument("--sample-size", type=int, default=0, help="0 means use all items")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-concurrency", type=int, default=1)
    parser.add_argument("--doc-concurrency", type=int, default=1)
    parser.add_argument("--backend", default="chatdoc", help="extractor backend name")
    parser.add_argument("--backend-entrypoint", default="", help="custom extractor function: module:function")
    parser.add_argument("--agent-log-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    args = parser.parse_args()

    data = _load_json(args.dataset)
    if not isinstance(data, list):
        raise ValueError("dataset must be a json list")

    if args.sample_size and args.sample_size > 0 and args.sample_size < len(data):
        import random

        rng = random.Random(args.seed)
        data = rng.sample(data, args.sample_size)

    agent = AsyncMutiDocAgent(
        log_dir=args.agent_log_dir,
        extractor_backend=args.backend,
        extractor_entrypoint=args.backend_entrypoint or None,
        doc_concurrency=args.doc_concurrency,
    )

    question_concurrency = max(1, int(args.question_concurrency or 1))
    question_semaphore = asyncio.Semaphore(question_concurrency)

    async def _run_one(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        async with question_semaphore:
            result = await run_single_query(agent, item, args.metadata_description)
        print(f"[{idx}/{len(data)}] success={result.get('success')} qid={result.get('qid')}")
        return {"idx": idx, "item": item, "result": result}

    jobs = [
        asyncio.create_task(_run_one(idx, item))
        for idx, item in enumerate(data, start=1)
    ]
    done = await asyncio.gather(*jobs) if jobs else []

    rows = []
    for payload in sorted(done, key=lambda x: int(x["idx"])):
        idx = int(payload["idx"])
        item = payload["item"]
        result = payload["result"]
        rows.append(
            {
                "index": idx,
                "question_id": result.get("qid"),
                "question": item.get("question"),
                "qid": result.get("qid"),
                "run_id": result.get("run_id"),
                "success": result.get("success"),
                "model_answer": result.get("result"),
                "json_data": result.get("json_data"),
                "json_dec": result.get("json_dec"),
                "code": result.get("code"),
                "code_resp": result.get("code_resp"),
                "agent_log_dir": result.get("log_dir"),
                "reference": {
                    "source_answer": item.get("source_answer"),
                    "final_answer": item.get("final_answer"),
                },
            }
        )

    _dump_json(args.output, rows)
    print(f"saved agent outputs: {args.output}")


if __name__ == "__main__":
    asyncio.run(cli_main())
