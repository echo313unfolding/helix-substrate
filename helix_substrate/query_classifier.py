"""
Lightweight query classifier for dual-model routing.

Routes queries to either TinyLlama (controller) or Qwen (coding).
Detects graph intent for FGIP tool routing.
Zero-GPU, keyword+regex based, sub-millisecond.

Based on the proven detect_code_intent() pattern from basin_server.
"""

import re
from enum import Enum
from typing import Dict, Optional, Tuple


class ModelTarget(Enum):
    TINYLLAMA = "tinyllama"
    QWEN_CODER = "qwen_coder"        # Default 1.5B coder
    QWEN_CODER_3B = "qwen_coder_3b"  # Upgraded 3B coder
    QWEN_INSTRUCT_3B = "qwen_instruct_3b"  # General 3B instruct (FGIP extraction, domain tasks)


class GraphIntent(Enum):
    SEARCH = "search"              # search_nodes
    EXPLORE = "explore"            # explore_connections
    CAUSAL = "causal"              # find_causal_chains
    THESIS = "thesis"              # get_thesis_score
    BOTH_SIDES = "both_sides"      # get_both_sides_patterns
    STATS = "stats"                # get_graph_stats


# Code-related keywords (any match → Qwen)
CODE_KEYWORDS = {
    # Explicit code tasks
    "write code", "write a function", "write a class", "write a script",
    "implement", "refactor", "rewrite this", "optimize this code",
    "code review", "review this code",
    # Bug fixing
    "fix this bug", "fix the bug", "debug this", "debug the",
    "find the bug", "what's wrong with this code",
    "fix this code", "fix the code", "fix this error",
    # Languages
    "python", "javascript", "typescript", "rust", "golang",
    "java", "c++", "sql", "bash script", "shell script",
    # Code constructs
    "function", "class", "method", "decorator", "async",
    "api endpoint", "rest api", "graphql",
    # File extensions in context
    ".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp",
}

# Regex patterns (any match → Qwen)
CODE_PATTERNS = [
    r"```",                          # Code fence
    r"^\s*(import|from)\s+\w+",      # Import statement
    r"^\s*(def|class|async def)\s+",  # Function/class definition
    r"def\s+\w+\s*\(",              # Function signature
    r"class\s+\w+[\s:(]",           # Class definition
    r"\w+\.\w+\(",                   # Method call pattern
    r"(if|for|while|try|except)\s",  # Control flow
    r"(return|yield|raise)\s",       # Return/yield/raise
    r"#\s*TODO|#\s*FIXME|#\s*BUG",  # Code comments
    r"\bsyntax error\b",            # Error types
    r"\btraceback\b",
    r"\bexception\b",
    r"\bindent(ation)?\s*(error)?",
]

# Controller keywords (strengthen TinyLlama routing when ambiguous)
CONTROLLER_KEYWORDS = {
    "what is", "how does", "explain", "describe", "tell me about",
    "status", "check", "list", "show me", "find",
    "search", "retrieve", "look up", "query",
    "component", "system", "architecture", "pipeline",
    "receipt", "proof", "verified", "proven",
    "compression ratio", "perplexity", "vram", "memory",
}


# Factual question patterns — suppress code routing when query is clearly factual.
# These override weak code signals (e.g., "python" as a language mention).
FACTUAL_QUESTION_PATTERNS = [
    r"^when (was|did|is|will)\b",
    r"^who (is|was|created|invented|founded|made)\b",
    r"^where (is|was|are)\b",
    r"^how (much|many|old|fast|slow|big|large|small|long|far)\b",
    r"^what (is|was|are) the (size|speed|price|cost|date|version|number|release)\b",
    r"\breleased?\b.*\b(when|date|year)\b",
    r"\b(when|date|year)\b.*\breleased?\b",
    r"\bhow many (parameters|layers|tokens|users|downloads)\b",
]

_compiled_factual_q = [re.compile(p, re.IGNORECASE) for p in FACTUAL_QUESTION_PATTERNS]


def _is_factual_question(query: str) -> bool:
    """Check if query is a factual/knowledge question, not a coding task."""
    for pat in _compiled_factual_q:
        if pat.search(query):
            return True
    return False


def classify(query: str) -> Tuple[ModelTarget, float, Dict]:
    """
    Classify a query for model routing.

    Returns:
        (target, confidence, debug_info)
        - target: ModelTarget.TINYLLAMA or ModelTarget.QWEN_CODER
        - confidence: 0.0-1.0
        - debug_info: dict with matched keywords/patterns
    """
    query_lower = query.lower().strip()

    code_keyword_hits = []
    code_pattern_hits = []
    controller_keyword_hits = []

    # Check code keywords
    for kw in CODE_KEYWORDS:
        if kw in query_lower:
            code_keyword_hits.append(kw)

    # Check code patterns
    for pat in CODE_PATTERNS:
        if re.search(pat, query, re.MULTILINE):
            code_pattern_hits.append(pat)

    # Check controller keywords
    for kw in CONTROLLER_KEYWORDS:
        if kw in query_lower:
            controller_keyword_hits.append(kw)

    # Score
    code_score = len(code_keyword_hits) * 0.3 + len(code_pattern_hits) * 0.5
    controller_score = len(controller_keyword_hits) * 0.3

    # Code fence is a strong signal
    if "```" in query:
        code_score += 1.0

    # Factual question override: if the query is clearly a knowledge question,
    # suppress weak code signals (e.g., bare language name mentions).
    # Only suppress if code signal is weak (just language keywords, no code patterns).
    is_factual = _is_factual_question(query)
    if is_factual and not code_pattern_hits:
        # Zero out code score — factual questions shouldn't route to coder
        # just because they mention a language name
        code_score = 0.0

    debug = {
        "code_keywords": code_keyword_hits,
        "code_patterns": code_pattern_hits,
        "controller_keywords": controller_keyword_hits,
        "code_score": round(code_score, 2),
        "controller_score": round(controller_score, 2),
        "factual_override": is_factual and not code_pattern_hits,
    }

    if code_score > 0 and code_score > controller_score:
        confidence = min(1.0, code_score / 2.0)
        return ModelTarget.QWEN_CODER, confidence, debug
    else:
        confidence = min(1.0, max(0.5, controller_score / 2.0))
        return ModelTarget.TINYLLAMA, confidence, debug


# ============================================================================
# GRAPH INTENT DETECTION
# ============================================================================

# Keyword sets per graph intent (checked in priority order)
_GRAPH_INTENT_KEYWORDS = {
    GraphIntent.THESIS: [
        "thesis score", "thesis", "verification score", "overall score",
        "how verified", "how proven",
    ],
    GraphIntent.BOTH_SIDES: [
        "both sides", "contradiction", "playing both", "same entity both",
        "problem and correction", "who benefits both",
    ],
    GraphIntent.CAUSAL: [
        "causal chain", "causal path", "how does .* lead to",
        "path from", "caused by", "leads to", "chain from",
        "what caused", "trace the path",
    ],
    GraphIntent.EXPLORE: [
        "connections", "connected to", "linked to", "network around",
        "who benefits", "explore", "edges for", "neighbors of",
        "what connects", "relationships of",
    ],
    GraphIntent.STATS: [
        "graph stats", "how many nodes", "how many edges",
        "graph size", "graph health", "graph status",
    ],
    GraphIntent.SEARCH: [
        "who is .* in the graph", "find .* in graph", "search graph",
        "graph search", "what do we know about", "in the graph",
        "in fgip", "graph node", "look up in graph",
    ],
}

# Entity name patterns — if the query mentions a known entity type alongside
# graph-suggestive words, boost graph intent detection
_GRAPH_ENTITY_PATTERNS = [
    r"\b(blackrock|vanguard|intel|tsmc|micron|nucor)\b",
    r"\b(chips act|genius act|fara|scotus|sec|fdic)\b",
    r"\b(lobbied|donated|funded|awarded grant|filed amicus)\b",
    r"\b(edge|node|claim|source tier|assertion)\b",
]
_compiled_entity_pats = [re.compile(p, re.IGNORECASE) for p in _GRAPH_ENTITY_PATTERNS]


def detect_graph_intent(query: str) -> Optional[GraphIntent]:
    """
    Detect if a query should route to FGIP graph tools.

    Returns the best-matching GraphIntent, or None if no graph intent detected.
    Checked in priority order: thesis > both_sides > causal > explore > stats > search.
    """
    query_lower = query.lower().strip()

    # Check keyword sets in priority order
    for intent, keywords in _GRAPH_INTENT_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, query_lower):
                return intent

    # If query mentions known entities AND has a question-like structure, default to SEARCH
    has_entity = any(p.search(query) for p in _compiled_entity_pats)
    has_question = any(w in query_lower for w in ("who", "what", "show", "tell me about", "find"))
    if has_entity and has_question:
        return GraphIntent.SEARCH

    return None
