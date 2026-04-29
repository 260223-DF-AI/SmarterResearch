"""
ResearchFlow — Input Guardrails Middleware

Detects and blocks prompt injection / stuffing attacks
in user inputs before they reach the agent pipeline.
"""

import re

BAD_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"you\s+are\s+now\s+(system|developer)",
    r"reveal\s+(the\s+)?(system|developer)\s+prompt",
    r"begin\s+new\s+instructions",
    r"act\s+as\s+.*(administrator|root|system)",
]


def detect_injection(user_input: str) -> bool:
    """
    Scan user input for common prompt injection patterns.

    - Check for system prompt override attempts.
    - Check for instruction stuffing patterns.
    - Return True if injection is detected, False otherwise.
    """

    lowered = user_input.lower()
    return any(re.search(pattern, lowered) for pattern in BAD_PATTERNS)


def sanitize_input(user_input: str) -> str:
    """
    Clean user input by removing or escaping dangerous patterns.

    - Strip known injection markers.
    - Escape special formatting that could manipulate prompts.
    - Return the sanitized string.
    """
    cleaned = user_input

    for pattern in BAD_PATTERNS:
        cleaned = re.sub(pattern, "[REMOVED]", cleaned, flags=re.IGNORECASE)

    # modify possibly manipulative characters
    cleaned = cleaned.replace("```", "` ` `")
    cleaned = cleaned.replace("{{", "{ {").replace("}}", "} }")

    return re.sub(r"\s+", " ", cleaned).strip()
