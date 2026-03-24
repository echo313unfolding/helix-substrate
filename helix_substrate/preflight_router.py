"""
Se-based preflight router for the AI OS.

Computes Symbolic Entropy (Se) on partial text input to predict the
route decision BEFORE the user submits.  Se is the real routing signal;
the final RouteDecision remains the authority on submit.

Se = log1p(H x C x D)  (all 0-1, continuous)

    H = Shannon entropy of character distribution (information density)
    C = semantic complexity (enriched with code/fact/graph markers)
    D = drift from recent queries (novelty)

Routing map (Se components -> AI OS decisions):
    c_code dominant   -> qwen_coder, code retrieval, code_128
    c_graph dominant  -> tinyllama, graph retrieval, graph_80
    c_fact dominant   -> tinyllama, exact_fact retrieval, factual_64
    otherwise         -> tinyllama, cache_first retrieval, factual_64

Lineage: se_lobe_router.py (H x C x D) + query_classifier.py (keywords)
         unified into a single Se-based preflight prediction.

Work Order: WO-AI-OS-PREFLIGHT-01 -> WO-AI-OS-SE-ROUTER-01
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .query_classifier import (
    ModelTarget,
    CODE_KEYWORDS,
    CODE_PATTERNS,
    CONTROLLER_KEYWORDS,
    FACTUAL_QUESTION_PATTERNS,
    detect_graph_intent,
)
from .route_decision import RetrievalMode


# ── Compiled patterns (built once at import) ──

_CODE_REGEXES = [re.compile(p, re.MULTILINE) for p in CODE_PATTERNS]
_FACTUAL_REGEXES = [re.compile(p, re.IGNORECASE) for p in FACTUAL_QUESTION_PATTERNS]

# Exact-fact start patterns (mirrors web_fact_tool.EXACT_FACT_PATTERNS)
_EXACT_FACT_STARTS = [
    "when was", "when did", "when is", "when will",
    "who is", "who was", "who created", "who invented", "who founded", "who made",
    "where is", "where was", "where are",
    "how much", "how many", "how old", "how tall", "how fast",
    "what is the population", "what is the capital",
    "what year", "what date",
]
_EXACT_FACT_REGEXES = [
    re.compile(r"^how much\b", re.IGNORECASE),
    re.compile(r"^how many\b", re.IGNORECASE),
    re.compile(r"^when (was|did|is|will)\b", re.IGNORECASE),
    re.compile(r"^who (is|was|created|invented|founded|made|built)\b", re.IGNORECASE),
    re.compile(r"^where (is|was|are)\b", re.IGNORECASE),
    re.compile(r"\bspecs?\b", re.IGNORECASE),
    re.compile(r"\bspecif?ications?\b", re.IGNORECASE),
]


# ── Se computation ──

@dataclass
class SeSignal:
    """Se components and derived routing signals for a text input."""
    # Core Se components (all 0-1)
    h: float       # Shannon entropy (information density)
    c: float       # Semantic complexity (composite)
    d: float       # Drift (novelty vs recent history)
    se: float      # log1p(H x C x D)

    # Sub-signals of C (what kind of complexity)
    c_code: float      # Code marker density (0-1)
    c_analytical: float  # Analytical/controller marker density (0-1)
    c_fact: float      # Exact-fact marker density (0-1)
    c_graph: float     # Graph intent marker density (0-1)
    c_creative: float  # Creative/open-ended marker density (0-1)

    # Factual question override (strong signal)
    factual_override: bool

    def dominant_signal(self) -> str:
        """Which sub-signal of C is dominant?"""
        signals = {
            "code": self.c_code,
            "graph": self.c_graph,
            "fact": self.c_fact,
            "analytical": self.c_analytical,
            "creative": self.c_creative,
        }
        best = max(signals, key=signals.get)
        # Only return dominant if it's meaningfully above zero
        if signals[best] < 0.05:
            return "neutral"
        return best

    def as_dict(self) -> dict:
        return {
            "se": round(self.se, 4),
            "h": round(self.h, 4),
            "c": round(self.c, 4),
            "d": round(self.d, 4),
            "c_code": round(self.c_code, 4),
            "c_analytical": round(self.c_analytical, 4),
            "c_fact": round(self.c_fact, 4),
            "c_graph": round(self.c_graph, 4),
            "c_creative": round(self.c_creative, 4),
            "factual_override": self.factual_override,
            "dominant": self.dominant_signal(),
        }


def compute_query_se(
    text: str,
    drift_history: Optional[List[str]] = None,
    graph_available: bool = False,
) -> SeSignal:
    """Compute Se for a query text, enriched for AI OS routing.

    This is the core Se computation that replaces pure keyword matching
    with a continuous entropy-based signal.  Sub-millisecond on any input.

    Args:
        text: Query text (possibly partial, during typing).
        drift_history: Recent query hashes for drift computation.
        graph_available: Whether FGIP graph is connected.

    Returns:
        SeSignal with all components and derived sub-signals.
    """
    if not text or not text.strip():
        return SeSignal(
            h=0, c=0, d=0, se=0,
            c_code=0, c_analytical=0, c_fact=0, c_graph=0, c_creative=0,
            factual_override=False,
        )

    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    words = set(text_lower.split())
    n_words = max(1, len(words))

    # ── H: Shannon entropy of character distribution ──
    char_counts: Dict[str, int] = {}
    for ch in text_lower:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    total_chars = len(text_lower)
    h = 0.0
    for count in char_counts.values():
        p = count / total_chars
        if p > 0:
            h -= p * math.log2(p)
    # Normalize: max Shannon entropy for printable ASCII ~ 6.6 bits
    h = min(1.0, h / 6.6)

    # ── C sub-signals: what kind of complexity ──

    # c_code: code keyword + pattern density
    code_kw_hits = sum(1 for kw in CODE_KEYWORDS if kw in text_lower)
    code_pat_hits = sum(1 for pat in _CODE_REGEXES if pat.search(text_stripped))
    # Code fence is a very strong signal
    if "```" in text_stripped:
        code_pat_hits += 3
    c_code = min(1.0, (code_kw_hits * 0.15 + code_pat_hits * 0.25))

    # c_analytical: controller/analytical keyword density
    ctrl_hits = sum(1 for kw in CONTROLLER_KEYWORDS if kw in text_lower)
    c_analytical = min(1.0, ctrl_hits * 0.15)

    # c_fact: exact-fact pattern density
    fact_start = any(text_lower.startswith(s) for s in _EXACT_FACT_STARTS)
    fact_regex = any(pat.search(text_stripped) for pat in _EXACT_FACT_REGEXES)
    c_fact = 0.5 if fact_start else (0.3 if fact_regex else 0.0)

    # c_graph: graph intent keyword density
    c_graph = 0.0
    if graph_available:
        graph_intent = detect_graph_intent(text_stripped)
        if graph_intent:
            c_graph = 0.8

    # c_creative: creative/open-ended markers (from se_lobe_router)
    _CREATIVE = {
        "imagine", "dream", "feel", "wonder", "create", "poetry", "soul",
        "meaning", "existence", "beautiful", "infinite", "emotion", "art",
    }
    creative_hits = len(words & _CREATIVE)
    c_creative = min(1.0, creative_hits * 0.2)

    # Factual question override: if query is clearly a knowledge question,
    # suppress code signal (e.g., "When was Python released?" should not
    # route to Qwen just because it mentions "Python").
    factual_override = False
    is_factual_q = any(pat.search(text_stripped) for pat in _FACTUAL_REGEXES)
    if is_factual_q and code_pat_hits == 0:
        c_code = 0.0
        factual_override = True

    # Composite C: weighted max of sub-signals (not sum — dominant signal wins)
    c = max(c_code, c_analytical, c_fact, c_graph, c_creative, 0.1)

    # ── D: Drift from recent queries ──
    text_hash = hashlib.sha256(text_stripped.encode()).hexdigest()[:12]
    if drift_history:
        overlap = sum(1 for rh in drift_history[-5:] if rh == text_hash)
        d = 1.0 - (overlap / min(5, len(drift_history)))
    else:
        d = 1.0  # First query = maximum novelty

    # ── Se = log1p(H x C x D) ──
    se = math.log1p(h * c * d)

    return SeSignal(
        h=h, c=c, d=d, se=se,
        c_code=c_code,
        c_analytical=c_analytical,
        c_fact=c_fact,
        c_graph=c_graph,
        c_creative=c_creative,
        factual_override=factual_override,
    )


# ── Data classes ──

@dataclass
class PreflightGuess:
    """Tentative route prediction from partial input, driven by Se."""
    # Routing predictions
    se_model_guess: str       # "tinyllama" or "qwen_coder"
    se_retrieval_guess: str   # "code", "exact_fact", "graph", "cache_first", "local"
    se_cache_guess: str       # "likely_hit", "possible", "unlikely"
    se_kv_guess: str          # "likely", "unlikely", "impossible"
    se_budget_guess: str      # "code_128", "factual_64", "graph_80"
    se_confidence: float      # 0.0-1.0

    # Se signal that produced these predictions
    se_signal: SeSignal

    # Metadata
    input_len: int
    compute_us: float

    def as_dict(self) -> dict:
        return {
            "se_model_guess": self.se_model_guess,
            "se_retrieval_guess": self.se_retrieval_guess,
            "se_cache_guess": self.se_cache_guess,
            "se_kv_guess": self.se_kv_guess,
            "se_budget_guess": self.se_budget_guess,
            "se_confidence": round(self.se_confidence, 3),
            "se": self.se_signal.as_dict(),
            "input_len": self.input_len,
            "compute_us": round(self.compute_us, 1),
        }

    # Compatibility aliases for comparison and REPL code
    @property
    def model_target(self) -> str:
        return self.se_model_guess

    @property
    def retrieval_mode(self) -> str:
        return self.se_retrieval_guess

    @property
    def cacheability(self) -> str:
        return self.se_cache_guess

    @property
    def kv_reuse(self) -> str:
        return self.se_kv_guess

    @property
    def budget_class(self) -> str:
        return self.se_budget_guess

    @property
    def confidence(self) -> float:
        return self.se_confidence


@dataclass
class PreflightComparison:
    """Comparison of Se preflight guess against final frozen route."""
    guess: dict
    final: dict
    model_match: bool
    retrieval_match: bool
    budget_match: bool
    overall_accuracy: float
    fields_matched: int
    fields_total: int
    early_warming_useful: bool
    warming_action: str

    def as_dict(self) -> dict:
        return {
            "preflight_guess": self.guess,
            "final_route": self.final,
            "model_match": self.model_match,
            "retrieval_match": self.retrieval_match,
            "budget_match": self.budget_match,
            "overall_accuracy": round(self.overall_accuracy, 3),
            "fields_matched": self.fields_matched,
            "fields_total": self.fields_total,
            "early_warming_useful": self.early_warming_useful,
            "warming_action": self.warming_action,
        }


@dataclass
class PreflightStats:
    """Cumulative accuracy statistics."""
    total: int = 0
    model_correct: int = 0
    retrieval_correct: int = 0
    budget_correct: int = 0
    warming_useful: int = 0

    def as_dict(self) -> dict:
        n = max(1, self.total)
        return {
            "total_queries": self.total,
            "model_accuracy": round(self.model_correct / n, 3),
            "retrieval_accuracy": round(self.retrieval_correct / n, 3),
            "budget_accuracy": round(self.budget_correct / n, 3),
            "warming_useful_rate": round(self.warming_useful / n, 3),
        }


# ── Se Router ──

class PreflightRouter:
    """Se-based incremental route predictor.

    Computes Se = log1p(H x C x D) on partial text input and maps the
    Se components to AI OS routing predictions (model, retrieval, cache,
    KV reuse, budget).

    The Se signal is continuous and stabilizes early — the dominant
    sub-signal of C (code vs fact vs graph vs analytical) determines
    the routing prediction as soon as enough characters arrive.

    Final routing authority remains RouteDecision (frozen on submit).

    Args:
        response_cache: ResponseCache instance (optional, for cacheability).
        kv_cache: KVPrefixCache instance (optional, for KV reuse).
        debounce_ms: Minimum interval between updates.
    """

    def __init__(
        self,
        response_cache=None,
        kv_cache=None,
        debounce_ms: float = 200.0,
    ):
        self.response_cache = response_cache
        self.kv_cache = kv_cache
        self.debounce_ms = debounce_ms
        self._last_guess: Optional[PreflightGuess] = None
        self._last_guess_time: float = 0.0
        self._drift_history: List[str] = []
        self.stats = PreflightStats()

    def guess(
        self,
        partial_text: str,
        current_model: str = "tinyllama",
        force_model: Optional[ModelTarget] = None,
        graph_available: bool = False,
    ) -> PreflightGuess:
        """Compute Se-based tentative route prediction from partial input.

        This is the hot path — must be sub-millisecond.
        """
        t0 = time.perf_counter()
        text = partial_text.strip()

        if not text:
            return PreflightGuess(
                se_model_guess="tinyllama",
                se_retrieval_guess="cache_first",
                se_cache_guess="unlikely",
                se_kv_guess="unknown",
                se_budget_guess="factual_64",
                se_confidence=0.0,
                se_signal=SeSignal(
                    h=0, c=0, d=0, se=0,
                    c_code=0, c_analytical=0, c_fact=0, c_graph=0, c_creative=0,
                    factual_override=False,
                ),
                input_len=0,
                compute_us=0.0,
            )

        # ── Compute Se ──
        se_sig = compute_query_se(
            text,
            drift_history=self._drift_history,
            graph_available=graph_available,
        )

        # ── Map Se components to routing decisions ──

        # Model target: forced > graph > code > default
        if force_model:
            model_target = force_model.value
        elif se_sig.c_graph > 0.5:
            model_target = "tinyllama"
        elif se_sig.c_code > 0.1 and se_sig.c_code > se_sig.c_analytical:
            model_target = "qwen_coder"
        else:
            model_target = "tinyllama"

        # Retrieval mode: graph > code > exact_fact > cache_first
        if se_sig.c_graph > 0.5:
            retrieval_mode = "graph"
        elif model_target == "qwen_coder":
            retrieval_mode = "code"
        elif se_sig.c_fact > 0.2:
            retrieval_mode = "exact_fact"
        else:
            retrieval_mode = "cache_first"

        # Budget class
        if se_sig.c_graph > 0.5:
            budget_class = "graph_80"
        elif model_target == "qwen_coder":
            budget_class = "code_128"
        else:
            budget_class = "factual_64"

        # Cacheability: from cache state
        cache_guess = self._predict_cacheability(text, model_target)

        # KV reuse: from model match + KV cache state
        kv_guess = self._predict_kv_reuse(model_target, current_model)

        # Confidence: Se magnitude × input length factor
        # Higher Se = more certain about the signal type.
        # Longer input = more characters to compute entropy from.
        len_factor = min(1.0, len(text) / 20)
        se_factor = min(1.0, se_sig.se / 0.3) if se_sig.se > 0 else 0.0
        dominant = se_sig.dominant_signal()
        # Strong dominant signal boosts confidence
        dom_boost = 1.0 if dominant != "neutral" else 0.5
        confidence = min(1.0, len_factor * max(se_factor, 0.3) * dom_boost)

        compute_us = (time.perf_counter() - t0) * 1_000_000

        guess = PreflightGuess(
            se_model_guess=model_target,
            se_retrieval_guess=retrieval_mode,
            se_cache_guess=cache_guess,
            se_kv_guess=kv_guess,
            se_budget_guess=budget_class,
            se_confidence=round(confidence, 3),
            se_signal=se_sig,
            input_len=len(text),
            compute_us=round(compute_us, 1),
        )

        self._last_guess = guess
        self._last_guess_time = time.perf_counter()
        return guess

    def record_query(self, query: str) -> None:
        """Record a submitted query for drift tracking."""
        text_hash = hashlib.sha256(query.strip().encode()).hexdigest()[:12]
        self._drift_history.append(text_hash)
        # Keep bounded
        if len(self._drift_history) > 50:
            self._drift_history = self._drift_history[-50:]

    def should_update(self) -> bool:
        """Debounce check: has enough time passed since last guess?"""
        if self._last_guess is None:
            return True
        elapsed_ms = (time.perf_counter() - self._last_guess_time) * 1000
        return elapsed_ms >= self.debounce_ms

    def compare(
        self,
        guess: PreflightGuess,
        final_route,  # RouteDecision
        actual_cache_hit: bool = False,
        actual_kv_event: str = "rebuilt",
    ) -> PreflightComparison:
        """Compare Se preflight guess against the real frozen route."""
        final_model = final_route.target_model.value
        final_retrieval = final_route.retrieval_mode.value

        if final_route.budget_mode == "graph":
            final_budget = "graph_80"
        elif final_route.max_tokens == 128:
            final_budget = "code_128"
        else:
            final_budget = "factual_64"

        model_match = guess.se_model_guess == final_model
        retrieval_match = guess.se_retrieval_guess == final_retrieval
        budget_match = guess.se_budget_guess == final_budget

        fields = [model_match, retrieval_match, budget_match]
        fields_matched = sum(fields)
        fields_total = len(fields)
        accuracy = fields_matched / fields_total

        # What could early warming have done?
        warming_action = "none"
        early_useful = False

        if model_match and guess.se_model_guess == "qwen_coder":
            warming_action = "preload_qwen"
            early_useful = True
        elif model_match and guess.se_retrieval_guess == "exact_fact" and retrieval_match:
            warming_action = "prestart_web"
            early_useful = True
        elif guess.se_cache_guess == "likely_hit" and actual_cache_hit:
            warming_action = "prefetch_cache"
            early_useful = True
        elif guess.se_kv_guess == "likely" and actual_kv_event == "reused":
            warming_action = "kv_ready"
            early_useful = True

        # Update stats
        self.stats.total += 1
        if model_match:
            self.stats.model_correct += 1
        if retrieval_match:
            self.stats.retrieval_correct += 1
        if budget_match:
            self.stats.budget_correct += 1
        if early_useful:
            self.stats.warming_useful += 1

        return PreflightComparison(
            guess=guess.as_dict(),
            final={
                "model_target": final_model,
                "retrieval_mode": final_retrieval,
                "budget_class": final_budget,
            },
            model_match=model_match,
            retrieval_match=retrieval_match,
            budget_match=budget_match,
            overall_accuracy=accuracy,
            fields_matched=fields_matched,
            fields_total=fields_total,
            early_warming_useful=early_useful,
            warming_action=warming_action,
        )

    def _predict_cacheability(self, text: str, model_target: str) -> str:
        """Predict whether the response cache would hit."""
        if self.response_cache is None:
            return "unknown"
        now = time.time()
        live_entries = sum(
            1 for entry in self.response_cache._cache.values()
            if now - entry.created_at < self.response_cache.ttl
        )
        if live_entries > 0:
            return "possible"
        return "unlikely"

    def _predict_kv_reuse(self, predicted_model: str, current_model: str) -> str:
        """Predict whether KV prefix cache would hit."""
        if predicted_model != current_model:
            return "impossible"
        if self.kv_cache is None:
            return "unknown"
        model_slots = sum(
            1 for state in self.kv_cache._slots.values()
            if state.model_target == predicted_model
        )
        if model_slots > 0:
            return "likely"
        return "unlikely"
