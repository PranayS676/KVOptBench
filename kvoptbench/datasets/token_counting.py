"""Minimal token-counting utilities for dataset preparation."""

from __future__ import annotations

from kvoptbench.client.timing import estimate_tokens


SUPPORTED_TOKEN_COUNT_METHODS = ("char_approx_4",)


def count_tokens(text: str, method: str = "char_approx_4") -> int:
    """Count tokens using supported local-only methods."""
    if method != "char_approx_4":
        raise ValueError(f"Unsupported token count method: {method}")
    return estimate_tokens(text)


def truncate_to_token_budget(text: str, token_budget: int, method: str = "char_approx_4") -> str:
    """Truncate text to an approximate token budget without tokenizer downloads."""
    if token_budget <= 0:
        return ""
    if count_tokens(text, method) <= token_budget:
        return text
    return text[: token_budget * 4].rstrip()
