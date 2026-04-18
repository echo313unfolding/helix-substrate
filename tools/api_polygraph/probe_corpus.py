"""Standardized probe corpus for model fingerprinting.

Each probe is designed to be:
- Deterministic at temperature=0
- Short (keep costs low)
- Diagnostic (different models produce different outputs)
- Diverse (exercises different model capabilities)

The corpus is fixed — never change it after fingerprinting, or
fingerprints become incompatible.
"""

# Version tag — bump if probes change (invalidates all fingerprints)
CORPUS_VERSION = "1"

# Each probe: (id, prompt, category, max_tokens)
# max_tokens kept small to minimize cost per probe
PROBES = [
    # Factual knowledge — model-specific phrasing differences
    ("fact_01", "The chemical formula for water is", "factual", 8),
    ("fact_02", "The speed of light in a vacuum is approximately", "factual", 12),
    ("fact_03", "The largest planet in our solar system is", "factual", 8),

    # Math — model-specific reasoning chains
    ("math_01", "What is 17 * 23? Answer with just the number:", "math", 4),
    ("math_02", "What is the square root of 144?", "math", 8),
    ("math_03", "If x = 3 and y = 7, what is x^2 + y?", "math", 8),

    # Code — highly model-specific formatting
    ("code_01", "Write a Python function that checks if a number is prime:\ndef is_prime(n):", "code", 48),
    ("code_02", "Complete this function:\ndef fibonacci(n):\n    if n <= 1:", "code", 32),
    ("code_03", "Write a one-liner to reverse a string in Python:", "code", 16),

    # Translation — model-specific phrasing
    ("trans_01", "Translate to French: 'The cat sat on the mat'", "translation", 16),
    ("trans_02", "Translate to Spanish: 'Good morning, how are you?'", "translation", 16),

    # Reasoning — different models structure arguments differently
    ("reason_01", "Is a hot dog a sandwich? Answer in one sentence.", "reasoning", 24),
    ("reason_02", "Why is the sky blue? Explain in one sentence.", "reasoning", 24),

    # Completion — model-specific continuations
    ("comp_01", "Once upon a time, in a land far away,", "completion", 24),
    ("comp_02", "The three most important inventions in human history are", "completion", 32),

    # Technical — domain-specific knowledge differences
    ("tech_01", "Explain what a buffer overflow is in one sentence:", "technical", 24),
    ("tech_02", "What is the difference between TCP and UDP?", "technical", 32),

    # Format following — models differ in how they follow instructions
    ("fmt_01", "List exactly 3 colors, one per line:", "format", 12),
    ("fmt_02", "Output only the word 'yes' or 'no': Is 7 a prime number?", "format", 4),

    # Edge case — low-probability tokens reveal model differences
    ("edge_01", "Continue this sequence: 1, 1, 2, 3, 5, 8, 13,", "edge", 8),
]


def get_probes(categories=None):
    """Get probe list, optionally filtered by category.

    Args:
        categories: list of category strings to include, or None for all

    Returns:
        List of (id, prompt, category, max_tokens) tuples
    """
    if categories is None:
        return list(PROBES)
    return [p for p in PROBES if p[2] in categories]


def get_probe_by_id(probe_id):
    """Get a single probe by ID."""
    for p in PROBES:
        if p[0] == probe_id:
            return p
    return None
