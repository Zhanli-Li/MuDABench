# MuDABench Dataset Release (Paper Supplement)

This folder is a **supplementary release package** for the paper:

**Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA**

It is **not a standalone project**. It provides the paper-aligned dataset artifacts used for benchmark evaluation and reproduction.

## Relation to the Paper

MuDABench studies analytical question answering over large, semi-structured multi-document collections.
This release corresponds to the benchmark setting described in the paper:

- total QA instances: 332
- split: `simple` and `complex` (166 each)
- average documents per question: 14.8
- document count in this release: 589 PDFs

## What Is Included

- `data/simple.json`
- `data/complex.json`
- `data/pdf/`

`data/pdf/<id>.pdf` is referenced by `metadata[].id` in `data/simple.json` / `data/complex.json`.

## Data Format (One Question)

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
  "final_answer": "reference final answer",
  "openai_vectors_id": "original evaluation-time vector store id"
}
```

Notes:

- `metadata` is the document-level structured evidence list for the question.
- `metadata[].schema` explains the semantics of `value_*` fields in that record.
- Different questions may use different subsets of `value_*` fields.

## Split Definition

- `data/simple.json`: questions requiring relatively lighter filtering/computation/reasoning chains.
- `data/complex.json`: questions requiring stronger conditional filtering, cross-document aggregation, and multi-step numerical/logical reasoning.

Both files follow the same schema.

## Release Sanitization

For open-source release, we removed:

- `chatdoc_upload_id`

All other fields were preserved to maintain evaluation consistency with the paper setting.

## Size Snapshot (2026-04-19)

- `data/pdf/` file count: 589
- total PDF size: ~3.87 GB
- largest PDF: ~56.07 MB

This package is intended for paper reproduction, so repository size is relatively large.

## Scope and Intended Use

This release is intended for:

- reproducing benchmark experiments
- evaluating multi-document analytical QA systems
- studying metadata-aware planning, extraction, normalization, and aggregation workflows

It is not intended as a general-purpose financial data product.

## Citation

If you use this dataset package, please cite the MuDABench paper.

```bibtex
@article{mudabench2026,
  title={Navigating Large-Scale Document Collections: MuDABench for Multi-Document Analytical QA},
  author={Li, Zhanli and Cao, Yixuan and Luo, Lvzhou and Luo, Ping},
  year={2026}
}
```

If your final BibTeX key/venue entry differs, replace this placeholder with the camera-ready citation.
