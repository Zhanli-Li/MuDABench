<div align="center">

# MuDABench

**Benchmark + reproducible toolkit for large-scale multi-document analytical QA**

[简体中文](./README_zh.md)

[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-b31b1b.svg)](https://2026.aclweb.org/)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Zhanli-Li/MuDABench)
[![License](https://img.shields.io/github/license/Zhanli-Li/MuDABench)](./LICENSE)

</div>

<p align="center">
  <img src="fig/case.png" alt="MuDABench case illustration" width="85%">
</p>

## Overview

**MuDABench** targets **multi-document analytical question answering** over large financial document collections (Chinese A-share + US market documents).

The primary metric promoted by this repository is `final_accuracy` (final answer accuracy), because even if we annotate intermediate atomic facts, the model may still arrive at the correct final answer through other atomic facts or different reasoning paths, which makes it difficult to compare the performance of different question-answering systems.


## What Is Included

| Component | Description |
|---|---|
| `data/simple.json` | 166 QA samples with concise final answers |
| `data/complex.json` | 166 QA samples with longer analytical final answers |
| `data/pdf/` | 589 source PDFs referenced by the QA samples |
| `agent/` | Reproducible multi-document agent (plan, per-doc extraction, normalization, code analysis, final answer) |
| `eval/evaluate.py` | Evaluator entry supporting `--eval-mode rag` and `--eval-mode agent` |
| `run_benchmark.py` | End-to-end runner (agent + evaluation) |

## Repository Structure

```text
MuDABench/
├── agent/
│   ├── agent_runner.py
│   ├── extractors.py
│   ├── ask_bchatdoc_adapter.py
│   ├── chatdoc_backend.py
│   └── extract_python_code.py
├── common/
│   ├── question_id.py
│   ├── openai_async_client.py
│   ├── json_utils.py
│   └── fake_backend.py
├── eval/
│   └── evaluate.py
├── prompts/
│   ├── agent_prompts.py
│   ├── eval_prompts.py
│   └── prompt_utils.py
├── data/
│   ├── simple.json
│   ├── complex.json
│   └── pdf/
├── run_benchmark.py
├── README.md
└── README_zh.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Common environment variables:

- `OPENROUTER_API_KEY`: required for agent and judge model calls
- `OPENROUTER_BASE_URL`: default `https://openrouter.ai/api/v1`
- `CHATDOC_API_KEY`: required when using default `chatdoc` backend
- `CHATDOC_BASE_URL`: default `https://api.chatdoc.studio`
- `JUDGE_MODEL`: default `deepseek/deepseek-v3.2`
- `JUDGE_TIMEOUT`: optional judge timeout (seconds), default handled in code

## Dataset Format

`--dataset` must be a JSON list. Each item should include question, metadata, and references.

```json
[
  {
    "question_id": "optional; auto-generated from question + metadata if missing",
    "question": "user question",
    "metadata": [
      {
        "symbol": "ticker or entity id",
        "year": 2021,
        "doctype": "document type",
        "chatdoc_upload_id": "document id (or doc_id/document_id/toBid/id)",
        "value_metric_name": 123.45,
        "detail_optional_field": "optional detail",
        "schema": {
          "value_metric_name": "metric description",
          "detail_optional_field": "field description"
        }
      }
    ],
    "source_answer": [
      "reference evidence / intermediate facts"
    ],
    "final_answer": "reference final answer"
  }
]
```

`question_id` rule (`common/question_id.py`):

```text
question_id = "q_" + sha256(json.dumps({"question": question, "metadata": metadata}, sort_keys=True))[:16]
```

Generate IDs for a dataset:

```bash
python common/question_id.py --dataset /path/to/dataset.json
```

## Workflow 1: Evaluate Classic RAG Outputs

For `--eval-mode rag`, `--agent-output` must be a JSON list and each item only allows these keys:

- `question_id` or `qid` (required)
- `retrieved_chunks` (required, list of strings)
- `model_answer` (required, string)

Example:

```json
[
  {
    "question_id": "q_9f4dd3a5f38b7f21",
    "retrieved_chunks": [
      "601398 2021 capital adequacy ratio is 18.02%.",
      "002142 2021 capital adequacy ratio is 15.44%."
    ],
    "model_answer": "601398, 002142, 601166"
  }
]
```

Run:

```bash
python eval/evaluate.py \
  --dataset /path/to/dataset.json \
  --agent-output /path/to/rag_outputs.json \
  --output-dir /path/to/rag_eval \
  --eval-mode rag \
  --sample-size 0 \
  --judge-concurrency 8
```

## Workflow 2: Run and Evaluate This Repository's Agent

End-to-end:

```bash
python run_benchmark.py \
  --dataset /path/to/dataset.json \
  --output-root /path/to/output_root \
  --sample-size 10 \
  --seed 42 \
  --question-concurrency 2 \
  --doc-concurrency 6 \
  --judge-concurrency 8
```

Step-by-step:

```bash
python agent/agent_runner.py \
  --dataset /path/to/dataset.json \
  --output /path/to/agent_output.json \
  --agent-log-dir /path/to/runs \
  --sample-size 10 \
  --seed 42 \
  --question-concurrency 2 \
  --doc-concurrency 6
```

```bash
python eval/evaluate.py \
  --dataset /path/to/dataset.json \
  --agent-output /path/to/agent_output.json \
  --output-dir /path/to/eval_dir \
  --eval-mode agent \
  --sample-size 10 \
  --seed 42 \
  --judge-concurrency 8
```

In `agent` mode, key output fields are:

- `question_id` or `qid`
- `agent_log_dir` or `log_dir` (must point to a directory containing `run_log.json`)
- `model_answer` or `result`

## Workflow 3: Evaluate Any Single-Document Agentic Method

If your method performs per-document extraction/QA, you can still use `--eval-mode agent` by preparing:

- `dataset.json` in the schema above
- `agent_output.json` aligned by `question_id`
- one `run_log.json` inside each `agent_log_dir`

Minimal `agent_output.json`:

```json
[
  {
    "question_id": "q_9f4dd3a5f38b7f21",
    "question": "What are the top companies by capital adequacy ratio in 2021?",
    "model_answer": "601398, 002142, 601166",
    "agent_log_dir": "/abs/path/to/custom_runs/q1"
  }
]
```

Minimal `run_log.json` requirement:

```json
{
  "doc_interactions": {
    "doc_601398_2021": [
      {
        "prompt": "Extract the 2021 capital adequacy ratio.",
        "answer": "The 2021 capital adequacy ratio is 18.02%."
      },
      {
        "field": "capital_adequacy_ratio_pct",
        "result": "18.02%"
      }
    ]
  }
}
```

`doc_interactions` keys must match metadata document IDs (`chatdoc_upload_id` / `doc_id` / `document_id` / `toBid` / `id`).

## Custom Extraction Backend

The agent backend is pluggable with this interface:

```python
extract_single_doc(document_id: str, prompt: str) -> str
```

Use `--backend-entrypoint module:function`.

Example with built-in fake backend:

```bash
python agent/agent_runner.py \
  --dataset /path/to/dataset.json \
  --output /path/to/fake_agent_output.json \
  --sample-size 1 \
  --backend-entrypoint common.fake_backend:extract_single_doc
```

## Evaluation Outputs and Metrics

`eval/evaluate.py` writes:

- `eval_summary.json`: aggregated metrics
- `eval_log.json`: per-question judge prompts, raw responses, parsed results

Main metrics:

- `final_accuracy` (**primary**): final answer correctness
- `cell_accuracy`: metric/detail value-level accuracy
- `info_accuracy`: alias of `row_accuracy`
- `row_accuracy`: whether each gold row is fully covered
- `column_accuracy`: field-level aggregated accuracy
- `solid_accuracy`: final answer correct and row coverage complete

Release note on fairness:

- In the final released version, we corrected about 10% annotation errors.
- Therefore, direct comparison between this release and numbers reported in the paper may be unfair.

## Links

- GitHub: <https://github.com/Zhanli-Li/MuDABench>
- Hugging Face Dataset: <https://huggingface.co/datasets/Zhanli-Li/MuDABench>

## Citation

If MuDABench is useful for your research, please cite:

```bibtex
@misc{mudabench2026,
  title        = {MuDABench: A Benchmark for Large-Scale Multi-Document Analysis},
  author       = {Li, Zhanli and others},
  year         = {2026},
  note         = {ACL 2026},
  howpublished = {\url{https://github.com/Zhanli-Li/MuDABench}}
}
```

## Ethical Considerations
The documents in MuDABench are collected from publicly available financial disclosures, ensuring that no private or non-public personal information is compromised. While we utilizes real-world financial figures, it is intended solely for the research.  We used AI for minor language polishing. To ensure accuracy, we work with the community to correct any potential annotation errors in the dataset on an ongoing basis; therefore, the evaluation results in our paper may not be up to date. 



## License

MuDABench is released under the **Apache License 2.0**. See [LICENSE](./LICENSE).
