"""
Lightweight post-generation quality gate.

Checks model output for common degenerate patterns before presenting
to user. Does NOT use an LLM — pure heuristic, <1ms.

Checks:
    1. Empty/whitespace-only output
    2. Repetition detection (n-gram loops)
    3. Overlong answer for simple queries
    4. Degenerate token patterns (encoding artifacts, repeated special tokens)

Work Order: WO-AI-OS-RUNTIME-01, Phase 4
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class QualityVerdict(Enum):
    PASS = "pass"
    EMPTY = "empty"
    REPETITION = "repetition"
    OVERLONG = "overlong"
    DEGENERATE = "degenerate"


@dataclass
class QualityResult:
    """Result of quality gate check."""
    verdict: QualityVerdict
    passed: bool
    details: str = ""
    suggestion: str = ""  # How to handle the failure

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "passed": self.passed,
            "details": self.details,
            "suggestion": self.suggestion,
        }


def check_quality(
    answer: str,
    query: str = "",
    max_tokens_generated: int = 0,
    is_code: bool = False,
) -> QualityResult:
    """Run all quality checks on generated output.

    Args:
        answer: Generated text.
        query: Original user query (for overlong check).
        max_tokens_generated: How many tokens were generated (0 = unknown).
        is_code: Whether this was a code generation request.

    Returns:
        QualityResult with verdict and details.
    """
    # 1. Empty/whitespace check
    stripped = answer.strip()
    if not stripped:
        return QualityResult(
            verdict=QualityVerdict.EMPTY,
            passed=False,
            details="Output is empty or whitespace-only",
            suggestion="retry_with_different_prompt",
        )

    # 2. Degenerate pattern check (before repetition, since these are worse)
    degen = _check_degenerate(stripped)
    if degen:
        return degen

    # 3. Repetition detection
    rep = _check_repetition(stripped)
    if rep:
        return rep

    # 4. Overlong check (only for non-code, simple queries)
    if not is_code and query:
        overlong = _check_overlong(stripped, query)
        if overlong:
            return overlong

    return QualityResult(
        verdict=QualityVerdict.PASS,
        passed=True,
        details=f"len={len(stripped)}, tokens~{max_tokens_generated}",
    )


def _check_degenerate(text: str) -> Optional[QualityResult]:
    """Check for degenerate output patterns."""
    # Repeated special tokens (e.g. <|endoftext|> repeated)
    if re.search(r'(<\|[^|]+\|>)\s*(\1\s*){3,}', text):
        return QualityResult(
            verdict=QualityVerdict.DEGENERATE,
            passed=False,
            details="Repeated special tokens detected",
            suggestion="retry_with_lower_temperature",
        )

    # Encoding artifacts (long runs of non-ASCII)
    if len(text) > 20:
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio < 0.3:
            return QualityResult(
                verdict=QualityVerdict.DEGENERATE,
                passed=False,
                details=f"Low ASCII ratio: {ascii_ratio:.2f}",
                suggestion="retry_with_different_prompt",
            )

    # All-punctuation or all-numbers
    if len(text) > 10 and not re.search(r'[a-zA-Z]{2,}', text):
        return QualityResult(
            verdict=QualityVerdict.DEGENERATE,
            passed=False,
            details="No alphabetic content",
            suggestion="retry_with_different_prompt",
        )

    return None


def _check_repetition(text: str) -> Optional[QualityResult]:
    """Detect n-gram repetition loops.

    Two complementary checks:
    1. High single n-gram frequency (any 3/5-gram > 30% of output)
    2. Low unique-gram ratio (< 40% unique 3-grams = cycling pattern)
    """
    words = text.lower().split()
    if len(words) < 12:
        return None  # Too short to meaningfully check

    from collections import Counter

    for n in (3, 5):
        if len(words) < n * 3:
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue

        counts = Counter(ngrams)
        most_common_ngram, most_common_count = counts.most_common(1)[0]
        frequency_ratio = most_common_count / len(ngrams)
        unique_ratio = len(counts) / len(ngrams)

        # Check 1: Any single n-gram dominates
        if frequency_ratio > 0.30:
            repeated = " ".join(most_common_ngram)
            return QualityResult(
                verdict=QualityVerdict.REPETITION,
                passed=False,
                details=f"{n}-gram '{repeated}' repeats {most_common_count}x "
                        f"({frequency_ratio:.1%} of output)",
                suggestion="truncate_at_first_repeat",
            )

        # Check 2: Low diversity (cycling short pattern)
        if unique_ratio < 0.15 and len(ngrams) > 20:
            return QualityResult(
                verdict=QualityVerdict.REPETITION,
                passed=False,
                details=f"Low {n}-gram diversity: {len(counts)} unique / {len(ngrams)} total "
                        f"({unique_ratio:.1%})",
                suggestion="truncate_at_first_repeat",
            )

    return None


def _check_overlong(text: str, query: str, threshold_ratio: float = 8.0) -> Optional[QualityResult]:
    """Check if answer is disproportionately long for a simple query.

    Simple heuristic: if answer is >8x longer than query (by word count)
    and query is short (< 10 words), flag as overlong.
    """
    query_words = len(query.split())
    answer_words = len(text.split())

    if query_words < 10 and answer_words > query_words * threshold_ratio and answer_words > 50:
        return QualityResult(
            verdict=QualityVerdict.OVERLONG,
            passed=False,
            details=f"Answer ({answer_words} words) is {answer_words/max(1,query_words):.1f}x "
                    f"longer than query ({query_words} words)",
            suggestion="truncate_or_summarize",
        )

    return None
