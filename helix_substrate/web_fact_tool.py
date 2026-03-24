"""
Web Fact Tool — Lightweight web search fallback for local assistant.

Architecture:
    Generic interface with swappable backends.
    Default: DuckDuckGo (instant answer API + lite search).
    Future: direct URL fetch, curated sources, targeted site search.

Usage:
    tool = WebFactTool()
    result = tool.search("NVIDIA Quadro T2000 VRAM")
    # result.text = "The Quadro T2000 has 4 GB GDDR5 VRAM..."
    # result.source = "techpowerup.com"
    # result.method = "ddg_lite+fetch"
"""

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WebFactResult:
    """A single fact retrieved from the web."""
    text: str
    source: str
    url: str = ""
    method: str = ""  # "ddg_instant", "ddg_lite", "ddg_lite+fetch", "url_fetch"
    latency_ms: float = 0.0


@dataclass
class WebSearchResult:
    """Full search result with metadata."""
    query: str
    facts: List[WebFactResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    backend: str = "duckduckgo"

    @property
    def best(self) -> Optional[str]:
        """Return the best fact text, or None."""
        return self.facts[0].text if self.facts else None

    def best_fact_passthrough(self) -> Optional[str]:
        """Format the best fact as a direct passthrough answer.

        Searches across ALL facts for the sentence that best answers
        the query, rather than blindly taking the first sentence of
        the first result.

        Returns a clean fact string suitable for direct display,
        or None if no good fact is available.
        """
        if not self.facts:
            return None

        def clean_html(text):
            text = text.replace("&#x27;", "'").replace("&amp;", "&")
            text = text.replace("&#92;", "\\").replace("&quot;", '"')
            text = text.replace("&#x2F;", "/").replace("&lt;", "<")
            text = text.replace("&gt;", ">")
            return text

        # Extract query keywords for relevance scoring
        stop_words = {"the", "a", "an", "is", "was", "are", "were", "has", "have",
                       "does", "do", "did", "how", "much", "many", "what", "when",
                       "who", "where", "which", "that", "this", "of", "in", "on",
                       "for", "with", "and", "or", "to", "it", "its"}
        keywords = [w.lower() for w in re.findall(r'\b\w+\b', self.query)
                    if w.lower() not in stop_words and len(w) > 2]

        # Synonym expansion for common measurement terms
        synonyms = {
            "vram": ["memory", "ram", "gddr", "hbm"],
            "ram": ["memory", "ddr"],
            "parameters": ["params", "trillion", "billion", "million"],
            "specs": ["specifications", "features"],
            "created": ["founded", "inventor", "author", "designed", "began", "started"],
            "creator": ["created", "founded", "inventor", "author", "designed", "began"],
            "released": ["release", "launched", "available"],
            "price": ["cost", "msrp"],
        }
        expanded = set(keywords)
        for kw in keywords:
            if kw in synonyms:
                expanded.update(synonyms[kw])

        # Detect if query asks for a quantity
        asks_quantity = bool(re.search(r'\bhow (much|many)\b', self.query, re.IGNORECASE))

        # Source preference: boost domains that are authoritative for query type
        source_boost = {}
        query_lower = self.query.lower()
        if any(w in query_lower for w in ("vram", "gpu", "spec", "memory", "ram",
                                           "cuda", "cores", "tdp", "watt", "clock")):
            # Hardware spec queries — prefer spec/reference sites
            for domain in ("techpowerup.com", "nvidia.com", "amd.com", "intel.com",
                           "notebookcheck.net", "anandtech.com", "tomshardware.com",
                           "videocardz.com", "hwinfo.com"):
                source_boost[domain] = 3.0
        elif any(w in query_lower for w in ("released", "release", "version", "date",
                                             "when was", "launched")):
            # Release/date queries — prefer official docs
            for domain in ("python.org", "github.com", "rust-lang.org", "nodejs.org",
                           "developer.mozilla.org", "docs.microsoft.com", "wiki"):
                source_boost[domain] = 2.0
        elif any(w in query_lower for w in ("who created", "who invented", "founder",
                                             "who made", "who built", "who designed")):
            # Creator/history queries — prefer encyclopedic + narrative sources
            for domain in ("wikipedia.org", "britannica.com", "wikidata.org",
                           "technologyreview.com", "hackernoon.com", "wired.com"):
                source_boost[domain] = 2.0
        elif any(w in query_lower for w in ("parameter", "param", "trillion", "billion",
                                             "layers", "tokens", "model size")):
            # Model spec queries — prefer research/reference
            for domain in ("arxiv.org", "huggingface.co", "wikipedia.org",
                           "paperswithcode.com"):
                source_boost[domain] = 2.0

        # Collect all sentences from all facts with their source
        candidates = []
        for fact in self.facts:
            text = clean_html(fact.text)
            # Split on sentence boundaries AND double-spaces / dashes common in spec pages
            sentences = re.split(r'(?<=[.!?])\s+|(?<=\S)\s{2,}(?=\S)|\s+-\s+-\s+|\s+-\s+(?=[A-Z])', text)
            source = fact.source
            if "/" in source:
                source = source.split("/")[0]
            for sent in sentences:
                sent = sent.strip().strip('-').strip()
                if len(sent) < 15:
                    continue
                sent_lower = sent.lower()

                # Score: keyword overlap (including synonyms)
                kw_score = sum(1 for kw in expanded if kw in sent_lower)

                # Bonus for sentences with number+unit (strong answer signal)
                has_number_unit = bool(re.search(
                    r'\d+\.?\d*\s*(gb|mb|tb|ghz|mhz|nm|watts?|kg|'
                    r'billion|trillion|million|thousand|percent|%|'
                    r'hours?|minutes?|seconds?|days?|years?|months?)',
                    sent_lower
                ))
                if has_number_unit:
                    kw_score += 2.0  # Heavy bonus for number+unit
                elif re.search(r'\d', sent):
                    kw_score += 0.5  # Small bonus for any number

                # Extra bonus when query asks "how much/many" and sentence has quantity
                if asks_quantity and has_number_unit:
                    kw_score += 1.0

                # Query-specific sentence bonuses
                if any(w in query_lower for w in ("who created", "who invented",
                                                   "who made", "who built", "creator")):
                    # Boost sentences with person-attribution patterns
                    if re.search(r'(created|designed|invented|founded|built|begun|began|started)\s+by\b', sent_lower):
                        kw_score += 4.0
                    elif re.search(r'\b(graydon|linus|guido|brendan|james|dennis|bjarne|ken|rob)\b', sent_lower):
                        kw_score += 3.0  # Known creator first names
                    elif re.search(r'(veteran|engineer|programmer|developer|researcher)', sent_lower):
                        kw_score += 2.0
                    # Penalize generic "X is a programming language" openings
                    if re.search(r'^.{0,30}(is a|is an)\s+(general|programming|compiled|interpreted)', sent_lower):
                        kw_score -= 2.0

                if any(w in query_lower for w in ("vram", "ram", "memory")) and asks_quantity:
                    # For memory capacity questions, heavily boost GB/MB answers
                    if re.search(r'\d+\s*(gb|mb)\b', sent_lower):
                        kw_score += 4.0

                # Penalize table-of-contents / feature-list sentences
                # (many commas, short clauses, no actual information)
                comma_count = sent.count(",")
                if comma_count > 6 and len(sent) > 100:
                    kw_score -= 3.0  # Heavy penalty for ToC-style listings

                # Penalize very short generic sentences
                if len(sent) < 30 and not has_number_unit:
                    kw_score -= 1.0

                # Penalize nav/chrome fragments (common in fetched pages)
                nav_words = ("home", "compare", "lists", "light", "menu", "login",
                             "sign up", "subscribe", "newsletter", "cookie")
                nav_count = sum(1 for w in nav_words if w in sent_lower)
                if nav_count >= 2:
                    kw_score -= 2.0

                # Penalize "bot check" / "JavaScript required" sentences
                if any(w in sent_lower for w in ("bot check", "javascript", "enable cookies", "captcha")):
                    kw_score -= 5.0

                # Source preference boost — authoritative domains for query type
                source_lower = source.lower()
                for domain, boost in source_boost.items():
                    if domain in source_lower:
                        kw_score += boost
                        break

                candidates.append((kw_score, sent, source))

        if not candidates:
            return None

        # Sort by relevance score, take best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_sent, best_source = candidates[0]

        # If the best sentence is very long (spec-page blob), try to extract
        # just the relevant window around the answer
        output = best_sent
        if len(output) > 200:
            # Find the position of the most relevant number+unit
            matches = list(re.finditer(
                r'(\d+\.?\d*\s*(?:gb|mb|tb|ghz|mhz|nm|tflops|watts?|billion|trillion|million))',
                output, re.IGNORECASE))
            # Pick the match with most keyword overlap nearby
            if matches:
                best_match = None
                best_mscore = -1
                for m in matches:
                    window_start = max(0, m.start() - 80)
                    window_end = min(len(output), m.end() + 80)
                    window = output[window_start:window_end].lower()
                    mscore = sum(1 for kw in expanded if kw in window)
                    if mscore > best_mscore:
                        best_mscore = mscore
                        best_match = m
                if best_match:
                    start = max(0, best_match.start() - 80)
                    end = min(len(output), best_match.end() + 80)
                    # Extend to word boundaries
                    while start > 0 and output[start - 1] not in ' \t\n':
                        start -= 1
                    while end < len(output) and output[end] not in ' \t\n':
                        end += 1
                    output = output[start:end].strip()
        if len(output) < 100 and len(candidates) > 1:
            _, second_sent, second_source = candidates[1]
            if second_source == best_source and len(second_sent) > 15:
                output += " " + second_sent

        # Truncate if too long
        if len(output) > 300:
            output = output[:297] + "..."

        return f"{output}\n  [Source: {best_source}]"

    def as_context(self, max_chars=600) -> str:
        """Format facts as prompt context."""
        if not self.facts:
            return ""
        parts = []
        chars = 0
        for f in self.facts:
            entry = f"[Web: {f.source}] {f.text}"
            if chars + len(entry) > max_chars:
                # Truncate to fit rather than skip entirely
                remaining = max_chars - chars
                if remaining > 50 and not parts:
                    # First entry: always include something
                    parts.append(entry[:remaining - 3] + "...")
                break
            parts.append(entry)
            chars += len(entry)
        return "\n".join(parts)


# --- Trigger heuristics ---

# Patterns that suggest the query needs external facts
FACTUAL_PATTERNS = [
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bwhat is the\b.*\b(size|speed|price|cost|date|version|number|amount|rate|percent)\b",
    r"\bhow (fast|slow|big|small|old|new|long|far)\b",
    r"\bwhen (was|did|is|will)\b",
    r"\bwho (is|was|created|invented|founded)\b",
    r"\bwhere (is|was|are)\b",
    r"\bcurrent(ly)?\b.*(status|version|price|state)",
    r"\blatest\b",
    r"\bspecif?ications?\b",
    r"\bspecs?\b",
    r"\b\d+\s*(gb|mb|tb|ghz|mhz|watts?|nm)\b",
    r"\brelease(d)?\s+(date|in|on)\b",
    r"\bcompare\b.*\bvs\b",
]

_compiled_factual = [re.compile(p, re.IGNORECASE) for p in FACTUAL_PATTERNS]

# Patterns for exact-answer questions (numbers, dates, names)
# These should use fact passthrough instead of LLM synthesis.
EXACT_FACT_PATTERNS = [
    r"^how much\b",
    r"^how many\b",
    r"\bhow (much|many) .*(have|has|does|do|did|is|are|was|were)\b",
    r"^when (was|did|is|will)\b",
    r"^who (is|was|created|invented|founded|made|built)\b",
    r"^where (is|was|are)\b",
    r"\b(how much|how many|what is the) (vram|ram|memory|storage|size|weight|price|cost)\b",
    r"\b(how many) (parameters?|layers?|tokens?|employees?|users?)\b",
    r"\brelease(d)?\s+(date|when|in|on)\b",
    r"\bspecs?\b",
    r"\bspecif?ications?\b",
]

_compiled_exact = [re.compile(p, re.IGNORECASE) for p in EXACT_FACT_PATTERNS]


def is_exact_fact_query(query: str) -> bool:
    """Check if query asks for an exact fact (number, date, name, spec).

    These should use fact passthrough to avoid LLM corruption of exact values.
    """
    for pat in _compiled_exact:
        if pat.search(query):
            return True
    return False


def needs_web_search(query: str, retrieval_score: float, is_code_query: bool = False) -> bool:
    """
    Decide whether web fallback should fire.

    Args:
        query: User query
        retrieval_score: Best score from Echo Memory (0.0 - 1.0)
        is_code_query: Whether classifier routed to coding model

    Returns:
        True if web search should be attempted
    """
    # Never use web for coding queries
    if is_code_query:
        return False

    # If local retrieval is strong, skip web
    if retrieval_score > 0.65:
        return False

    # If query matches factual patterns, always try web
    for pat in _compiled_factual:
        if pat.search(query):
            return True

    # If retrieval is weak, try web for anything non-trivial
    if retrieval_score < 0.35 and len(query.split()) >= 3:
        return True

    return False


# --- DuckDuckGo backend ---

class DuckDuckGoBackend:
    """DuckDuckGo search via instant answer API + lite HTML search."""

    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    TIMEOUT = 8

    def instant_answer(self, query: str) -> Optional[WebFactResult]:
        """Try DDG instant answer API first (fastest, most structured)."""
        t0 = time.time()
        try:
            url = (
                "https://api.duckduckgo.com/?q="
                + urllib.parse.quote(query)
                + "&format=json&no_html=1&skip_disambig=1"
            )
            resp = urllib.request.urlopen(url, timeout=self.TIMEOUT)
            data = json.loads(resp.read())

            parts = []
            if data.get("Abstract"):
                parts.append(data["Abstract"])
            if data.get("Answer"):
                parts.append(data["Answer"])
            # Add top related topics
            for t in data.get("RelatedTopics", [])[:2]:
                if isinstance(t, dict) and "Text" in t:
                    parts.append(t["Text"])

            if parts:
                source = data.get("AbstractSource", "DuckDuckGo")
                source_url = data.get("AbstractURL", "")
                return WebFactResult(
                    text="\n".join(parts)[:500],
                    source=source,
                    url=source_url,
                    method="ddg_instant",
                    latency_ms=(time.time() - t0) * 1000,
                )
        except Exception:
            pass
        return None

    def lite_search(self, query: str, max_results: int = 3) -> List[dict]:
        """Search DDG lite and extract titles, snippets, URLs.

        DDG lite HTML structure: results are in <td> elements.
        Pattern: title at TD[3], snippet at TD[5], URL at TD[7],
        then next result at TD[11], TD[13], TD[15], etc. (stride of 8).
        """
        try:
            url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=self.TIMEOUT)
            html = resp.read().decode("utf-8", errors="replace")

            # Extract all <td> contents
            tds = re.findall(r"<td[^>]*>(.*?)</td>", html, re.DOTALL)

            def clean_td(raw):
                text = re.sub(r"<[^>]+>", "", raw).strip()
                text = re.sub(r"&\w+;", " ", text)
                return re.sub(r"\s+", " ", text).strip()

            results = []
            # Title TDs appear at indices 3, 11, 19, 27, ... (stride 8)
            # Snippet = title_idx + 2, URL = title_idx + 4
            for title_idx in range(3, len(tds), 8):
                if len(results) >= max_results:
                    break
                snippet_idx = title_idx + 2
                url_idx = title_idx + 4

                if url_idx >= len(tds):
                    break

                title = clean_td(tds[title_idx])
                snippet = clean_td(tds[snippet_idx])
                result_url = clean_td(tds[url_idx])

                if title and len(title) > 5:
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": result_url,
                    })

            return results
        except Exception:
            return []

    def fetch_page_snippet(self, url: str, query: str, max_chars: int = 500) -> Optional[str]:
        """Fetch a page and extract relevant text snippet."""
        try:
            if not url.startswith("http"):
                url = "https://" + url

            req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
            resp = urllib.request.urlopen(req, timeout=self.TIMEOUT)
            html = resp.read().decode("utf-8", errors="replace")[:80000]

            # Strip scripts, styles, tags
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            # Reject bot-check / JS-only pages
            if any(marker in text.lower() for marker in
                   ("bot check in progress", "javascript must be enabled",
                    "enable javascript", "captcha", "checking your browser")):
                return None

            # Find most relevant section using query keywords
            keywords = [w.lower() for w in query.split() if len(w) > 3]
            best_pos = 0
            best_score = 0
            window = 200

            for pos in range(0, min(len(text), 10000), 50):
                chunk = text[pos : pos + window].lower()
                score = sum(1 for kw in keywords if kw in chunk)
                if score > best_score:
                    best_score = score
                    best_pos = pos

            start = max(0, best_pos - 50)
            return text[start : start + max_chars].strip()
        except Exception:
            return None


class WebFactTool:
    """
    Generic web fact tool with swappable backends.

    Current backends:
        - duckduckgo (default): instant answer API + lite search + page fetch
    """

    def __init__(self, backend: str = "duckduckgo"):
        self.backend_name = backend
        if backend == "duckduckgo":
            self._backend = DuckDuckGoBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def search(self, query: str, fetch_pages: bool = True) -> WebSearchResult:
        """
        Search for facts about the query.

        Pipeline:
            1. DDG instant answer (fastest, ~1s)
            2. DDG lite search (get titles + snippets, ~2s)
            3. Fetch top result page for deeper context (optional, ~2s)

        Args:
            query: Search query
            fetch_pages: Whether to fetch full pages for better snippets

        Returns:
            WebSearchResult with facts and metadata
        """
        t0 = time.time()
        result = WebSearchResult(query=query, backend=self.backend_name)

        # Tier 1: Instant answer
        instant = self._backend.instant_answer(query)
        if instant:
            result.facts.append(instant)

        # Tier 2: Lite search
        search_results = self._backend.lite_search(query)
        for sr in search_results:
            if sr["snippet"] and len(sr["snippet"]) > 20:
                result.facts.append(WebFactResult(
                    text=sr["snippet"],
                    source=sr["url"],
                    url=sr["url"],
                    method="ddg_lite",
                ))

        # Tier 3: Fetch top page for deeper snippets when:
        # - We have few/no good snippets, OR
        # - Query asks "how much/many" but no snippet contains a number+unit
        needs_fetch = (not result.facts or len(result.facts[0].text) < 100)
        if not needs_fetch and re.search(r'\bhow (much|many)\b', query, re.IGNORECASE):
            # Check if any snippet has a relevant number+unit answer.
            # Extract what the query asks about to match relevant units.
            query_lower = query.lower()
            if any(w in query_lower for w in ("vram", "ram", "memory", "storage")):
                relevant_units = r'(gb|mb|tb)'
            elif any(w in query_lower for w in ("parameter", "param")):
                relevant_units = r'(billion|trillion|million)'
            elif any(w in query_lower for w in ("speed", "fast", "clock", "frequency")):
                relevant_units = r'(ghz|mhz|tok/s)'
            else:
                relevant_units = r'(gb|mb|tb|ghz|mhz|billion|trillion|million|percent|%)'

            has_relevant_quantity = any(
                re.search(r'\d+\.?\d*\s*' + relevant_units, f.text, re.IGNORECASE)
                for f in result.facts
            )
            if not has_relevant_quantity:
                needs_fetch = True

        if fetch_pages and search_results and needs_fetch:
            # Try fetching up to 3 result pages until we get a good snippet
            for sr in search_results[:3]:
                fetch_url = sr["url"]
                t_fetch = time.time()
                snippet = self._backend.fetch_page_snippet(fetch_url, query)
                if snippet and len(snippet) > 50:
                    result.facts.insert(0 if not instant else 1, WebFactResult(
                        text=snippet,
                        source=fetch_url,
                        url=fetch_url,
                        method="ddg_lite+fetch",
                        latency_ms=(time.time() - t_fetch) * 1000,
                    ))
                    break

        result.total_latency_ms = (time.time() - t0) * 1000
        return result


# --- CLI test ---

if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NVIDIA Quadro T2000 VRAM"
    print(f"Query: {query}\n")

    tool = WebFactTool()
    result = tool.search(query)

    print(f"Backend: {result.backend}")
    print(f"Latency: {result.total_latency_ms:.0f}ms")
    print(f"Facts: {len(result.facts)}\n")

    for i, fact in enumerate(result.facts):
        print(f"[{i+1}] ({fact.method}, {fact.latency_ms:.0f}ms)")
        print(f"    Source: {fact.source}")
        print(f"    {fact.text[:300]}")
        print()

    print("--- Context for LLM ---")
    print(result.as_context())
