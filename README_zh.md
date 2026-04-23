<div align="center">

# MuDABench

**面向大规模多文档分析问答的 Benchmark 与可复现工具链**

[![ACL 2026 Findings](https://img.shields.io/badge/ACL%202026-Findings-b31b1b.svg)](https://2026.aclweb.org/)
[![Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-ffcc00?logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Zhanli-Li/MuDABench)
[![License](https://img.shields.io/github/license/Zhanli-Li/MuDABench)](./LICENSE)

</div>

<p align="center">
  <img src="fig/case.png" alt="MuDABench case illustration" width="85%">
</p>

## 概述

**MuDABench** 聚焦于金融场景（中国 A 股 + 美股文档）下的**多文档分析问答**。

本仓库主推的指标是 `final_accuracy`（最终答案准确率），因为即使我们标注了中间原子事实，模型也可能通过其他原子事实或不同推理路径得到正确最终答案，这使得不同问答系统的评估存在不可对比性。

## 仓库内容

| 组件 | 说明 |
|---|---|
| `data/simple.json` | 166 条简洁最终答案样本 |
| `data/complex.json` | 166 条长分析最终答案样本 |
| `data/pdf/` | 589 份被样本引用的原始 PDF |
| `agent/` | 可复现 agent（规划、单文档抽取、归一化、代码分析、最终回答） |
| `eval/evaluate.py` | 评测入口，支持 `--eval-mode rag` 与 `--eval-mode agent` |
| `run_benchmark.py` | 端到端执行（agent + evaluate） |

## 项目结构

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

## 环境准备

安装依赖：

```bash
pip install -r requirements.txt
```

常用环境变量：

- `OPENROUTER_API_KEY`：agent 与 judge 调用模型时需要
- `OPENROUTER_BASE_URL`：默认 `https://openrouter.ai/api/v1`
- `CHATDOC_API_KEY`：使用默认 `chatdoc` backend 时需要
- `CHATDOC_BASE_URL`：默认 `https://api.chatdoc.studio`
- `JUDGE_MODEL`：默认 `deepseek/deepseek-v3.2`
- `JUDGE_TIMEOUT`：judge 超时秒数，可选

## 数据集格式

`--dataset` 必须是 JSON list。每条样本需要包含问题、文档元信息和参考答案。

```json
[
  {
    "question_id": "可选；缺失时按 question + metadata 自动生成",
    "question": "用户问题",
    "metadata": [
      {
        "symbol": "实体标识（如股票代码）",
        "year": 2021,
        "doctype": "文档类型",
        "chatdoc_upload_id": "文档 id（也可用 doc_id/document_id/toBid/id）",
        "value_metric_name": 123.45,
        "detail_optional_field": "可选细节字段",
        "schema": {
          "value_metric_name": "指标说明",
          "detail_optional_field": "字段说明"
        }
      }
    ],
    "source_answer": [
      "参考证据/中间事实"
    ],
    "final_answer": "最终答案参考"
  }
]
```

`question_id` 规则（`common/question_id.py`）：

```text
question_id = "q_" + sha256(json.dumps({"question": question, "metadata": metadata}, sort_keys=True))[:16]
```

可用以下命令生成数据集 id：

```bash
python common/question_id.py --dataset /path/to/dataset.json
```

## 场景一：评测传统 RAG 中间结果

`--eval-mode rag` 下，`--agent-output` 必须是 JSON list，且每条记录只允许以下字段：

- `question_id` 或 `qid`（必填）
- `retrieved_chunks`（必填，字符串列表）
- `model_answer`（必填，字符串）

示例：

```json
[
  {
    "question_id": "q_9f4dd3a5f38b7f21",
    "retrieved_chunks": [
      "601398 2021 年资本充足率为 18.02%。",
      "002142 2021 年资本充足率为 15.44%。"
    ],
    "model_answer": "601398、002142、601166"
  }
]
```

运行命令：

```bash
python eval/evaluate.py \
  --dataset /path/to/dataset.json \
  --agent-output /path/to/rag_outputs.json \
  --output-dir /path/to/rag_eval \
  --eval-mode rag \
  --sample-size 0 \
  --judge-concurrency 8
```

## 场景二：运行并评测本仓库 Agent

端到端运行：

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

分步运行：

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

`agent` 模式下关键输出字段：

- `question_id` 或 `qid`
- `agent_log_dir` 或 `log_dir`（目录内需要有 `run_log.json`）
- `model_answer` 或 `result`

## 场景三：评测任意单文档 Agentic 方法

如果你的方法核心是“按文档进行抽取/问答”，无需改评测代码，只需准备：

- `dataset.json`（遵循上面的 schema）
- `agent_output.json`（按 `question_id` 对齐）
- 每个 `agent_log_dir` 下放一个 `run_log.json`

最小 `agent_output.json` 示例：

```json
[
  {
    "question_id": "q_9f4dd3a5f38b7f21",
    "question": "What are the top companies by capital adequacy ratio in 2021?",
    "model_answer": "601398、002142、601166",
    "agent_log_dir": "/abs/path/to/custom_runs/q1"
  }
]
```

最小 `run_log.json` 要求：

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

`doc_interactions` 的 key 必须与 metadata 中文档 id 对齐（`chatdoc_upload_id` / `doc_id` / `document_id` / `toBid` / `id`）。

## 自定义抽取后端

agent 支持可插拔后端，统一接口：

```python
extract_single_doc(document_id: str, prompt: str) -> str
```

通过 `--backend-entrypoint module:function` 注入。

示例（仓库内 fake backend）：

```bash
python agent/agent_runner.py \
  --dataset /path/to/dataset.json \
  --output /path/to/fake_agent_output.json \
  --sample-size 1 \
  --backend-entrypoint common.fake_backend:extract_single_doc
```

## 评测输出与指标

`eval/evaluate.py` 会写出：

- `eval_summary.json`：聚合指标
- `eval_log.json`：逐题 judge prompt、raw response、解析结果

核心指标：

- `final_accuracy`（**主推**）：最终答案正确率
- `cell_accuracy`：指标值/细节值级准确率
- `info_accuracy`：`row_accuracy` 的别名
- `row_accuracy`：每个 gold row 是否完整覆盖
- `column_accuracy`：字段级聚合准确率
- `solid_accuracy`：最终答案正确且 row 全覆盖

版本与公平性说明：

- 在最终发布版本中，我们修正了约 10% 的标注错误。
- 因此，将本发布版结果与论文中报告的对比结果直接比较，可能不完全公平。

## 链接

- GitHub: <https://github.com/Zhanli-Li/MuDABench>
- Hugging Face 数据集: <https://huggingface.co/datasets/Zhanli-Li/MuDABench>

## 引用

如果 MuDABench 对你的研究有帮助，欢迎引用：

```bibtex
@misc{mudabench2026,
  title        = {MuDABench: A Benchmark for Large-Scale Multi-Document Analysis},
  author       = {Li, Zhanli and others},
  year         = {2026},
  note         = {ACL 2026},
  howpublished = {\url{https://github.com/Zhanli-Li/MuDABench}}
}
```

## 许可证

MuDABench 使用 **Apache License 2.0**，详见 [LICENSE](./LICENSE)。
