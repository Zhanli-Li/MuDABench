---
pretty_name: MuDABench
license: apache-2.0
language:
- zh
task_categories:
- question-answering
size_categories:
- n<1K
tags:
- question-answering
- multi-document
- finance
- chinese
---

# MuDABench

MuDABench is a benchmark for multi-document analytical question answering over large-scale document collections.

Repository links:

- Hugging Face dataset: https://huggingface.co/datasets/Zhanli-Li/MuDABench
- GitHub repository: https://github.com/Zhanli-Li/MuDABench

## Overview

This release contains:

- `data/simple.json`: 166 QA samples with concise final answers.
- `data/complex.json`: 166 QA samples with more detailed analytical final answers.
- `data/pdf/`: 589 source PDF files referenced by the samples.

The benchmark is centered on analytical QA over Chinese A-share market documents. Each sample requires aggregating information across multiple documents instead of reading a single source in isolation.

## Data Format

Each item in `data/simple.json` or `data/complex.json` is a multi-document analytical QA sample:

```json
{
  "question": "...",
  "metadata": [
    {
      "id": "uuid-used-as-pdf-filename",
      "symbol": "company ticker",
      "year": 2021,
      "doctype": "document type",
      "schema": {
        "value_xxx": "field meaning"
      },
      "value_xxx": "structured value"
    }
  ],
  "source_answer": "intermediate supporting facts (text)",
  "final_answer": "reference final answer"
}
```

Notes:

- `metadata` is the document-level structured evidence list for the question.
- `metadata[].id` matches the PDF filename stem in `data/pdf/`.
- `metadata[].schema` explains the semantics of the `value_*` fields in that record.
- Different questions may use different subsets of `value_*` fields.
- The public release does not include `openai_vectors_id`.

## File Structure

```text
MuDABench/
├── data/
│   ├── simple.json
│   ├── complex.json
│   └── pdf/
├── LICENSE
└── README.md
```

## Intended Use

MuDABench is intended for:

- evaluating multi-document analytical QA systems
- testing retrieval plus reasoning pipelines over document collections
- benchmarking Chinese financial document QA workflows

## License

MuDABench is released under the Apache License 2.0. See `LICENSE` for details.
