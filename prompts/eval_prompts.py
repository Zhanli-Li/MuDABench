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

RAG_INFO_DOUBLE_CHECK_CORRECT_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
Prompt: Judge — RAG Information Extraction (The correct side)

The user will provide:
1. Information (`list`) needed to answer the question (total_required = {len_source}):
{source_answer}

2. Information extracted by the model (each part separated by <next_chunk> is independent of each other):
{info}

Evaluation Criteria:
1. If the information extracted by the model contains the key information with true entity from the reference, the correct extraction is added by one.
2. If the information does not match the entity or does not provide the entity information, it is judged incorrect and the number of correct extractions remains unchanged.
3. If the key information is missing or does not match the reference, it is judged incorrect and the number of correct extractions remains unchanged.
4. If the extraction is missing or does not match the reference information, it is judged incorrect and the number of correct extractions remains unchanged.
5. If the extracted information involves numerical values, the model is considered correct as long as it correctly extracts integer digits and the first decimal place.

Note: If the extraction of the model contains information other than the reference information or uses a different language, this does not affect the determination. Multiple correct extractions of the same information are counted only once.

TASK: Calculate the number of intersections between the information extracted by the model and the information that needs to be extracted.

Return JSON only:
```json
{{
    "correct_extractions": <number of correct entries>,
    "total_required": {len_source},
    "explanation": "short explanation"
}}
```
"""
    ).strip()
)

RAG_INFO_DOUBLE_CHECK_INCORRECT_PROMPT = PromptTemplate(
    template=textwrap.dedent(
        """
Prompt: Judge — RAG Information Extraction (The incorrect side)

The user will provide:
1. Information (`list`) needed to answer the question (total_required = {len_source}):
{source_answer}

2. Information extracted by the model (each part separated by <next_chunk> is independent of each other):
{info}

Evaluation Criteria:
1. Treat the reference information list as the gold standard.
2. For each required entry in the reference list, check all extracted chunks.
3. If at least one chunk correctly contains the key information with the true entity, this entry is counted as correctly extracted.
4. If the key information is missing, conflicts with the reference (wrong entity/value), or is otherwise incorrect, this entry is counted as an error.
5. If the extracted information involves numerical values, the model is considered correct as long as it correctly extracts integer digits and the first decimal place.
6. Extra information that is not in the reference list, or uses a different language, does not affect the judgment. Only the correctness of the required entries is considered.
7. Multiple correct extractions of the same reference entry are counted only once.
8. error_extractions is the number of required entries that are incorrect or missing (i.e., entries in the reference list that do not have a correct extraction).

TASK: You are asked to consider each part of the model output separated by <next_chunk>, as a piece of information extracted by the model, and compute:
1. error_extractions = number of incorrect or missing entries among the required information.
2. total_required = {len_source}.

Return JSON only:
```json
{{
    "error_extractions": <number of incorrect or missing entries>,
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
