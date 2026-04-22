"""
Default adapter for ChatDOC single-document extraction.

This module intentionally mirrors the role of the original `agent/ask_bchatdoc.py`,
but it is replaceable.
"""

from agent.chatdoc_backend import ask_chatdoc

__all__ = ["ask_chatdoc"]
