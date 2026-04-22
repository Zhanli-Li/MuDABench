import importlib
from typing import Any, Callable, Dict, Optional

from agent.ask_bchatdoc_adapter import ask_chatdoc


SingleDocExtractor = Callable[[str, str], str]


class ExtractorRegistry:
    """
    Registry for pluggable single-document extraction backends.

    Core contract:
    - input: document_id (str), prompt (str)
    - output: extracted answer text (str)

    This makes `ask_bchatdoc.py` replaceable by any single-doc extractor system.
    """

    _registry: Dict[str, SingleDocExtractor] = {}

    @classmethod
    def register(cls, name: str, func: SingleDocExtractor) -> None:
        cls._registry[name] = func

    @classmethod
    def get(cls, name: str) -> SingleDocExtractor:
        if name not in cls._registry:
            raise KeyError(f"unknown extractor backend: {name}")
        return cls._registry[name]

    @classmethod
    def load_from_entrypoint(cls, spec: str) -> SingleDocExtractor:
        """
        `spec` format: module.submodule:function_name
        Example: my_backend.chat:extract_single_doc
        """
        if ":" not in spec:
            raise ValueError("entrypoint must be module_path:function_name")
        module_path, func_name = spec.split(":", 1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(f"entrypoint {spec} is not callable")
        return func


def _chatdoc_backend(document_id: str, prompt: str) -> str:
    return ask_chatdoc(document_id=document_id, prompt=prompt)


ExtractorRegistry.register("chatdoc", _chatdoc_backend)


def resolve_extractor(backend: str = "chatdoc", backend_entrypoint: Optional[str] = None) -> SingleDocExtractor:
    if backend_entrypoint:
        return ExtractorRegistry.load_from_entrypoint(backend_entrypoint)
    return ExtractorRegistry.get(backend)
