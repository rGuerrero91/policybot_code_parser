"""Shared utilities for inference methods."""
import re

from src.policybot_hcpcs.models.schemas import TextSpan

# Patterns for HCPCS/CPT codes
_CODE_PATTERNS = [
    re.compile(r'\b([A-Z]\d{4})\b'),   # e.g., J3031, G0008
    re.compile(r'\b(\d{5})\b'),          # e.g., 99213 (5-digit CPT)
]

CONTEXT_CHARS = 30


def find_literal_codes(text: str, known_codes: set[str]) -> dict[str, TextSpan]:
    """
    Scan text for literal HCPCS/CPT code patterns and return those
    that exist in the known_codes set.

    Returns a dict of code -> TextSpan (first occurrence only).
    """
    matches: dict[str, TextSpan] = {}
    for pattern in _CODE_PATTERNS:
        for match in pattern.finditer(text):
            code = match.group(1)
            if code in known_codes and code not in matches:
                start, end = match.span(1)
                ctx_start = max(0, start - CONTEXT_CHARS)
                ctx_end = min(len(text), end + CONTEXT_CHARS)
                matches[code] = TextSpan(
                    start=start,
                    end=end,
                    text=text[ctx_start:ctx_end],
                )
    return matches
