import textwrap
from prompts.prompt_utils import PromptTemplate

INFO_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
The user will provide:
- Information (`list`) needed to answer the question (total_required = {len_source}): {source_answer}
- Information extracted by the model : {agent_conversation}

Evaluation Criteria:
- If the information extracted by the model contains the key information with true entity from the reference, the correct extraction is added by one.
- If the information doesn't match the entity or doesn't provide the entity information, it is incorrect.
- If key information is missing or does not match the reference, it is incorrect.
- For numeric values, extraction is correct if integer digits and first decimal place are correct.

Note: Extra information or different language does not hurt correctness. Duplicate correct extraction counts once.

Return JSON only:
```json
{{
    "correct_extractions": <number>,
    "total_required": {len_source},
    "explanation": "short explanation"
}}
```
"""
    ).strip()
)

ROW_METRIC_ONLY_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
The user will provide:
- Multi-document question: {question}
- Source table headers:
{source_headers}
- One gold source row (`row_index = {row_index}`):
{source_row}
- Aligned target document metadata (already matched by the dataset, do not judge metadata again):
{aligned_doc_meta}
- Metric columns to judge from this row (`metric_total = {metric_total}`):
{metric_columns}
- Complete dialog record of all sub-question answers on this document:
{agent_conversation}

Evaluation criteria:
- Judge only against this single gold row and this single document dialog.
- This gold row has already been aligned to the correct document by dataset metadata.
- Do not score metadata fields in this step. Judge only metric columns one by one.
- A metric column is correct only if dialog contains the same core fact under correct symbol/year context.
- For numeric columns, integer digits + first decimal place are enough unless strict identifiers are required.
- Extra information does not hurt correctness.

Return JSON only:
```json
{{
  "correct_metric_fields": ["<metric field name>"]
}}
```
"""
    ).strip()
)

FINAL_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
The user will provide:
- Multi-document question: {question}
- Multi-document reference answer: {final_answer}
- Model's answer: {chatdoc_answer}

Evaluation Criteria:
- Model answer is correct if it contains key information in reference answer.
- If key information is missing or mismatched, it is incorrect.
- For numeric values, integer digits + first decimal place are enough.

Return JSON only:
```json
{{
    "is_correct": true/false,
    "explanation": "short explanation"
}}
```
"""
    ).strip()
)
