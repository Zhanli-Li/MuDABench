def extract_single_doc(document_id: str, prompt: str) -> str:
    return f"[fake-backend] doc={document_id}; extracted_from_prompt={prompt[:80]}"
