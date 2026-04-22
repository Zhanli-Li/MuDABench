# MuDABench Agent

本仓库包含一个可复现的多文档 agent pipeline，以及一个可复用的评测入口。评测入口以 `question_id` 对齐 dataset 和方法输出，支持三类常见场景：

- 评测传统 RAG 的中间检索结果：使用 `--eval-mode rag`。
- 运行并评测本仓库的 agent：先跑 `agent/agent_runner.py`，再用 `--eval-mode agent` 评测。
- 评测任意一种聚焦于单文档的 agentic 方法：把该方法的单文档问答轨迹整理成本仓库兼容的 `run_log.json`，再用 `--eval-mode agent` 评测。

## 项目结构

- `agent/`
  - `agent/agent_runner.py`: Agent 主流程（规划、单文档抽取、结构化、代码分析、最终回答）
  - `extractors.py`: 抽取后端注册与动态加载（支持 `module:function`）
  - `ask_bchatdoc_adapter.py`: 默认 ChatDOC 适配层
  - `chatdoc_backend.py`: ChatDOC API 交互与 app cache 逻辑
  - `extract_python_code.py`: 从 LLM 输出中提取可执行 Python 代码
- `eval/`
  - `eval/evaluate.py`: 评测入口（`agent` / `rag` 两种模式）
- `prompts/`
  - `agent_prompts.py`: Agent 侧提示词
  - `eval_prompts.py`: 评测侧提示词
  - `prompt_utils.py`: 提示词模板工具
- `common/`
  - `openai_async_client.py`: OpenAI/兼容端异步客户端封装
  - `json_utils.py`: 鲁棒 JSON 解析工具
  - `fake_backend.py`: 示例后端（用于验证可插拔接口）
- `data/`
  - `easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json`
  - `hard_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json`
  - `chatdoc_app_cache.json`
- `run_benchmark.py`: 本仓库 agent 的端到端串联执行（Agent + agent-mode Eval）
- `run_single_case.sh`: 1 条样本快速 smoke test

## 环境变量

常用：

- `OPENROUTER_API_KEY`: agent 和 judge 调用 OpenRouter/兼容 OpenAI 接口时需要。
- `OPENROUTER_BASE_URL`: 默认 `https://openrouter.ai/api/v1`。
- `CHATDOC_API_KEY`: 使用默认 chatdoc backend 时必需。
- `CHATDOC_BASE_URL`: 默认 `https://api.chatdoc.studio`。
- `JUDGE_MODEL`: 默认 `deepseek/deepseek-v3.2`。
- `JUDGE_TIMEOUT`: judge 单次调用超时时间，默认由代码兜底。
- `CHATDOC_APP_CACHE_PATH`: 可选，默认 `data/chatdoc_app_cache.json`。

## 数据集格式

`eval/evaluate.py` 的 `--dataset` 参数必须指向一个 JSON list。每个 item 至少需要包含问题、参考答案和文档元信息。

### question_id 规则

每条样本的唯一 id 由 `question + metadata` 稳定生成。生成逻辑在 `common/question_id.py` 中，等价于：

```text
question_id = "q_" + sha256(json.dumps({"question": question, "metadata": metadata}, sort_keys=True))[:16]
```

如果 dataset item 已经显式提供 `question_id`，评测会优先使用该值；否则自动按 `question + metadata` 生成。所有方法输出都必须带 `question_id` 或兼容字段 `qid`。

可以用下面命令为 dataset 生成 id 清单：

```bash
python common/question_id.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json
```

### Dataset Schema

```json
[
  {
    "question_id": "可选；不提供时由 question + metadata 自动生成",
    "question": "用户问题",
    "metadata": [
      {
        "symbol": "文档所属实体，例如股票代码",
        "year": 2021,
        "doctype": "文档类型",
        "chatdoc_upload_id": "文档 id，可替换为 doc_id/document_id/toBid/id",
        "value_metric_name": 123.45,
        "detail_optional_field": "可选的其他待评测字段",
        "schema": {
          "value_metric_name": "指标说明",
          "detail_optional_field": "字段说明"
        }
      }
    ],
    "source_answer": [
      "传统 RAG 模式下用于 info judge 的参考证据/答案条目"
    ],
    "final_answer": "最终答案参考"
  }
]
```

说明：

- `metadata` 中的文档 id 会按 `chatdoc_upload_id`、`doc_id`、`document_id`、`toBid`、`id` 的顺序识别。
- 数据集不要提供额外表格字段。评测会从 `metadata` 中带有 `value_` / `detail_` 前缀的字段还原出内部临时表，再基于这张表做 row/cell judge。

## 场景一：评测传统 RAG 中间结果

传统 RAG 不需要提供 agent 日志。你只需要把检索出来的 chunks 和最终答案整理成严格的 `agent-output` 格式，然后用 `--eval-mode rag`。

### RAG agent-output Schema

`--agent-output` 必须是 JSON list。每条记录必须通过 `question_id` 与 dataset 中的样本对齐，且 RAG 模式只接受以下字段：

- `question_id`: 必填。由 `question + metadata` 生成，也可以使用兼容字段 `qid`。
- `retrieved_chunks`: 必填。字符串列表，表示传统 RAG 检索到的中间结果。
- `model_answer`: 必填。RAG 系统基于 chunks 生成的最终答案。

最小示例：

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

注意：不要把 RAG chunks 放在 dataset 里。当前约定要求所有被评测方法的输出都在 `agent-output` 中显式提供。

### 运行命令

```bash
python eval/evaluate.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json \
  --agent-output /path/to/rag_outputs.json \
  --output-dir /path/to/rag_eval \
  --eval-mode rag \
  --sample-size 0 \
  --judge-concurrency 8
```

注意：

- `agent-output` 会按 `question_id` 与 dataset 对齐，不依赖 list 顺序。
- `retrieved_chunks` 必须是字符串列表，不能是 dict list 或混合类型。
- `model_answer` 会用于 `final_accuracy`；`retrieved_chunks` 会用于 RAG 中间结果的 info judge。

## 场景二：运行并评测本仓库的 Agent

本仓库的 agent 会先对每个文档做单文档问答，再汇总成结构化 JSON、执行分析代码并生成最终答案。评测时会读取 agent 运行目录下的 `run_log.json`，对每个文档、每个 row 的指标抽取做 judge。

### 端到端运行

```bash
python run_benchmark.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json \
  --output-root tmp_smoke \
  --sample-size 10 \
  --seed 42 \
  --question-concurrency 2 \
  --doc-concurrency 6 \
  --judge-concurrency 8
```

输出：

- `tmp_smoke/agent_output.json`
- `tmp_smoke/eval/eval_summary.json`
- `tmp_smoke/eval/eval_log.json`

### 分步运行 Agent

```bash
python agent/agent_runner.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json \
  --output tmp_smoke/agent_output.json \
  --agent-log-dir tmp_smoke/runs \
  --sample-size 10 \
  --seed 42 \
  --question-concurrency 2 \
  --doc-concurrency 6
```

然后单独评测：

```bash
python eval/evaluate.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json \
  --agent-output tmp_smoke/agent_output.json \
  --output-dir tmp_smoke/eval \
  --eval-mode agent \
  --sample-size 10 \
  --seed 42 \
  --judge-concurrency 8
```

### 本仓库 agent-output 示例

`agent/agent_runner.py` 会自动生成如下结构：

```json
[
  {
    "index": 1,
    "question_id": "q_9f4dd3a5f38b7f21",
    "question": "针对中国A股市场，请你给出你知识库中2021年资本充足率最高的三家公司股票代码",
    "qid": "q_9f4dd3a5f38b7f21",
    "run_id": "uuid",
    "success": true,
    "model_answer": "601398、002142、601166",
    "json_data": "[{\"symbol\":\"601398\",\"capital_adequacy_ratio_pct\":18.02}]",
    "json_dec": "字段说明",
    "code": "用于分析的 Python 代码",
    "code_resp": "代码执行结果",
    "agent_log_dir": "tmp_smoke/runs/<qid>/<run_id>",
    "reference": {
      "source_answer": [
        "根据2021年数据，股票代码601398的资本充足率为18.02%"
      ],
      "final_answer": "根据2021年数据分析，资本充足率最高的三家公司股票代码分别是：601398、002142、601166。"
    }
  }
]
```

评测 `--eval-mode agent` 时，关键字段是：

- `question_id` / `qid`: 与 dataset 中由 `question + metadata` 生成的 id 对齐。
- `agent_log_dir` / `log_dir`: 指向包含 `run_log.json` 的目录。
- `model_answer` / `result`: 用于最终答案 judge。

## 场景三：评测任意单文档 Agentic 方法

如果你的方法不是本仓库的 agent，但它的流程是“围绕每个单文档进行一次或多次问答/抽取”，可以不改评测代码，只需要把每个文档 id 上的全部信息抽取过程或抽取结果整理进兼容的 `run_log.json`。评测会把 `metadata` 中的 `value_` / `detail_` 字段自动展开成 gold rows，再按文档 id 拉取该文档上的所有抽取过程作为上下文，对 `values.*`、`details.*` 等列逐列 judge。

### 数据准备

你需要准备两类文件：

- `dataset.json`: 使用上面的 dataset schema，且每个 metadata entry 都要包含可识别的文档 id，以及用于评测的 `value_` / `detail_` 字段。
- `agent_output.json`: JSON list，每个 item 通过 `question_id` 指向该题对应的日志目录。

`agent_output.json` 最小示例：

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

### run_log.json Schema

在每个 `agent_log_dir` 下放一个 `run_log.json`。场景三只需要 `doc_interactions` 字段：第一层 key 是文档 id，value 是一个 list，list 里放这个文档上的信息抽取结果或单文档对话记录。

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
    ],
    "doc_002142_2021": [
      {
        "prompt": "Extract the 2021 capital adequacy ratio.",
        "answer": "The 2021 capital adequacy ratio is 15.44%."
      },
      {
        "field": "capital_adequacy_ratio_pct",
        "result": "15.44%"
      }
    ]
  }
}
```

字段说明：

- `doc_interactions` 的第一层 key 必须是文档 id，并且要能和 dataset metadata 的 `chatdoc_upload_id`、`doc_id`、`document_id`、`toBid` 或 `id` 对上。
- `doc_interactions[doc_id]` 必须是 list。这个 list 表示从该文档上抽取出来的所有信息，或围绕该文档发生的单文档对话记录。
- 推荐每个 list item 使用 `{"prompt": "...", "answer": "..."}` 表示一轮单文档问答。
- 如果你的方法直接输出结构化抽取结果，也可以用 `{"field": "...", "result": "..."}`、`{"content": "..."}`、`{"text": "..."}`，甚至直接放字符串。评测会把它们统一拼成该文档的上下文。
- 同一个文档 id 下可以放多个抽取结果。评测会把这个文档上的所有条目拼成 `Q: ... A: ...` 的文档级上下文。
- 对每个由 metadata 展开的 gold row，评测先找到对应文档，再把该文档上下文和该 row 的 `values.*` / `details.*` 列交给 judge，逐列判断哪些字段被正确抽取。

### 运行命令

```bash
python eval/evaluate.py \
  --dataset /path/to/dataset.json \
  --agent-output /path/to/custom_agent_output.json \
  --output-dir /path/to/custom_agent_eval \
  --eval-mode agent \
  --sample-size 0 \
  --judge-concurrency 8
```

注意：

- 这种方式评测的是“某个单文档 id 上的信息抽取过程是否覆盖该文档对应的 metadata gold row 和每个字段列”，不是评测全局检索 chunks。
- 如果你的方法只返回全局 chunks，没有按文档组织问答轨迹，应使用 `--eval-mode rag`。
- 如果你的方法有最终答案，把它写入 `model_answer` 或 `result`，这样会同时得到 `final_accuracy`。

## 自定义抽取后端

本仓库 agent 的单文档抽取后端可插拔，统一接口为：

```python
extract_single_doc(document_id: str, prompt: str) -> str
```

可通过 `--backend-entrypoint module:function` 注入。

示例（使用仓库内置 fake backend）：

```bash
python agent/agent_runner.py \
  --dataset data/easy_all_data_with_independent_openai_vector_store_with_tobid_revise_with_atomic_metadata.json \
  --output tmp_smoke/fake_agent_output.json \
  --sample-size 1 \
  --backend-entrypoint common.fake_backend:extract_single_doc
```

## 快速 Smoke Test

```bash
bash run_single_case.sh
```

或指定数据集：

```bash
bash run_single_case.sh /abs/path/to/dataset.json
```

## 评测输出

`eval/evaluate.py` 会输出：

- `eval_summary.json`: 聚合指标，包括 `row_accuracy`、`column_accuracy`、`cell_accuracy`、`final_accuracy`、`solid_accuracy`。
- `eval_log.json`: 每题的 judge prompt、raw response、解析结果和中间日志，便于排查误判。

指标含义：

- `row_accuracy`: gold row 是否被方法完整覆盖。
- `column_accuracy`: 按字段聚合的准确率。
- `cell_accuracy`: 具体指标值/细节字段的准确率。
- `final_accuracy`: 最终答案是否正确。
- `solid_accuracy`: 同时满足最终答案正确且 row 全覆盖。

## 说明

- `__init__.py` 为空是正常的，用于标记包目录，便于导入和工具链识别。
- 当前评测实现要求 `agent-output` 中每条记录包含 `question_id` 或 `qid`。如果缺失，评测会报错，避免结果与 dataset 错配。
