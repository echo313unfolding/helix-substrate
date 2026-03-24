"""
FGIP Graph Bridge — Direct SQLite access to the FGIP knowledge graph.

Provides sync tool wrappers for local_assistant.py to call without
requiring the MCP server. Reads directly from fgip.db.

Tools exposed:
  - search_nodes       — FTS/LIKE search on nodes
  - explore_connections — BFS N-hop edge traversal
  - find_causal_chains  — Follow causal edge types via BFS
  - get_thesis_score   — Problem/correction edge counts + both-sides
  - get_graph_stats    — Node/edge/claim/source counts by type
  - get_both_sides     — Entities on both problem and correction sides
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from helix_substrate.query_classifier import GraphIntent

FGIP_DB = Path.home() / "fgip-engine" / "fgip.db"

# Edge type classifications (mirror mcp_server.py)
PROBLEM_EDGE_TYPES = (
    "LOBBIED_FOR", "DONATED_TO", "FUNDED_BY", "REGISTERED_AS_AGENT",
    "FILED_AMICUS", "EMPLOYED", "OWNS_MEDIA", "HAS_LEVERAGE_OVER",
    "BLOCKS", "HOLDS_TREASURY", "OWNS", "INVESTED_IN",
)
CORRECTION_EDGE_TYPES = (
    "AWARDED_GRANT", "BUILT_IN", "FUNDED_PROJECT", "IMPLEMENTED_BY",
    "RULEMAKING_FOR", "AUTHORIZED_BY", "CORRECTS", "ENABLES",
    "REDUCES", "FUNDS", "CONTRIBUTES_TO", "RECEIVED_FUNDING",
)
CAUSAL_EDGE_TYPES = (
    "CAUSED", "ENABLED", "TRIGGERED", "LEADS", "CONTRIBUTES_TO",
    "BENEFITS_FROM", "CORRECTS", "BLOCKS", "REDUCES", "INCREASES_RISK_FOR",
    "DEPENDS_ON", "CONTROLS", "DETERMINES", "COMPLICATES",
)


class FGIPBridge:
    """Direct SQLite bridge to the FGIP knowledge graph."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or FGIP_DB
        self._available = self.db_path.exists()

    @property
    def available(self) -> bool:
        return self._available

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def search_nodes(self, term: str, node_type: str = None, limit: int = 10) -> dict:
        """Search nodes by name/description/id."""
        conn = self._conn()
        try:
            like = f"%{term}%"
            if node_type:
                rows = conn.execute(
                    "SELECT node_id, name, node_type, description FROM nodes "
                    "WHERE (name LIKE ? OR description LIKE ? OR node_id LIKE ?) "
                    "AND node_type = ? LIMIT ?",
                    (like, like, like, node_type.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT node_id, name, node_type, description FROM nodes "
                    "WHERE name LIKE ? OR description LIKE ? OR node_id LIKE ? LIMIT ?",
                    (like, like, like, limit),
                ).fetchall()
            return {"count": len(rows), "nodes": [dict(r) for r in rows]}
        finally:
            conn.close()

    def explore_connections(self, node_id: str, depth: int = 1) -> dict:
        """BFS edge traversal around a node."""
        depth = min(depth, 3)
        conn = self._conn()
        try:
            # Verify node exists (try exact, then partial)
            node = conn.execute(
                "SELECT node_id, name, node_type FROM nodes WHERE node_id = ?",
                (node_id,),
            ).fetchone()
            if not node:
                partials = conn.execute(
                    "SELECT node_id FROM nodes WHERE node_id LIKE ? LIMIT 5",
                    (f"%{node_id}%",),
                ).fetchall()
                if partials:
                    return {"error": f"Node not found. Did you mean: {[r['node_id'] for r in partials]}"}
                return {"error": f"Node not found: {node_id}"}

            visited = {node_id}
            connections = []
            current_level = [node_id]

            for d in range(depth):
                next_level = []
                for nid in current_level:
                    for edge in conn.execute(
                        "SELECT edge_id, edge_type, from_node_id, to_node_id, assertion_level, confidence "
                        "FROM edges WHERE from_node_id = ?", (nid,),
                    ).fetchall():
                        connections.append({"depth": d + 1, "dir": "->", "edge": dict(edge)})
                        tid = edge["to_node_id"]
                        if tid not in visited:
                            visited.add(tid)
                            next_level.append(tid)

                    for edge in conn.execute(
                        "SELECT edge_id, edge_type, from_node_id, to_node_id, assertion_level, confidence "
                        "FROM edges WHERE to_node_id = ?", (nid,),
                    ).fetchall():
                        connections.append({"depth": d + 1, "dir": "<-", "edge": dict(edge)})
                        fid = edge["from_node_id"]
                        if fid not in visited:
                            visited.add(fid)
                            next_level.append(fid)

                current_level = next_level

            return {
                "node": dict(node),
                "total_connections": len(connections),
                "connections": connections[:60],  # cap for LLM context
            }
        finally:
            conn.close()

    def find_causal_chains(self, start_node: str = None, end_node: str = None, max_depth: int = 4) -> dict:
        """Follow causal edge types via BFS. Lightweight alternative to ReasoningAgent."""
        conn = self._conn()
        try:
            placeholders = ",".join("?" * len(CAUSAL_EDGE_TYPES))

            if start_node and end_node:
                # BFS from start to end along causal edges
                queue = [(start_node, [start_node])]
                visited = {start_node}
                chains = []
                while queue and len(chains) < 5:
                    current, path = queue.pop(0)
                    if len(path) > max_depth + 1:
                        continue
                    rows = conn.execute(
                        f"SELECT to_node_id, edge_type FROM edges "
                        f"WHERE from_node_id = ? AND edge_type IN ({placeholders})",
                        (current, *CAUSAL_EDGE_TYPES),
                    ).fetchall()
                    for row in rows:
                        next_id = row["to_node_id"]
                        edge_type = row["edge_type"]
                        new_path = path + [f"--{edge_type}-->", next_id]
                        if next_id == end_node:
                            chains.append(new_path)
                        elif next_id not in visited:
                            visited.add(next_id)
                            queue.append((next_id, new_path))
                return {"start": start_node, "end": end_node, "chains_found": len(chains), "chains": chains}

            elif start_node:
                # Fan out from start node along causal edges
                visited = {start_node}
                paths = []
                queue = [(start_node, [start_node], 0)]
                while queue and len(paths) < 10:
                    current, path, depth = queue.pop(0)
                    if depth >= max_depth:
                        paths.append(path)
                        continue
                    rows = conn.execute(
                        f"SELECT to_node_id, edge_type FROM edges "
                        f"WHERE from_node_id = ? AND edge_type IN ({placeholders})",
                        (current, *CAUSAL_EDGE_TYPES),
                    ).fetchall()
                    if not rows:
                        paths.append(path)
                    for row in rows:
                        next_id = row["to_node_id"]
                        edge_type = row["edge_type"]
                        new_path = path + [f"--{edge_type}-->", next_id]
                        if next_id not in visited:
                            visited.add(next_id)
                            queue.append((next_id, new_path, depth + 1))
                return {"start": start_node, "paths_found": len(paths), "paths": paths}

            else:
                # No seed — find top causal hubs
                rows = conn.execute(
                    f"SELECT from_node_id, COUNT(*) as cnt FROM edges "
                    f"WHERE edge_type IN ({placeholders}) "
                    f"GROUP BY from_node_id ORDER BY cnt DESC LIMIT 10",
                    CAUSAL_EDGE_TYPES,
                ).fetchall()
                return {"causal_hubs": [{"node_id": r["from_node_id"], "outgoing_causal": r["cnt"]} for r in rows]}
        finally:
            conn.close()

    def get_thesis_score(self) -> dict:
        """Problem/correction edge counts + both-sides detection."""
        conn = self._conn()
        try:
            p_ph = ",".join("?" * len(PROBLEM_EDGE_TYPES))
            c_ph = ",".join("?" * len(CORRECTION_EDGE_TYPES))

            problem_count = conn.execute(
                f"SELECT COUNT(*) FROM edges WHERE edge_type IN ({p_ph})", PROBLEM_EDGE_TYPES
            ).fetchone()[0]
            correction_count = conn.execute(
                f"SELECT COUNT(*) FROM edges WHERE edge_type IN ({c_ph})", CORRECTION_EDGE_TYPES
            ).fetchone()[0]
            total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            total_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

            # Both-sides entities
            both_sql = """
            SELECT DISTINCT e1.from_node_id
            FROM edges e1 JOIN edges e2 ON e1.from_node_id = e2.from_node_id
            WHERE e1.edge_type IN ({}) AND e2.edge_type IN ({})
            """.format(p_ph, c_ph)
            both_sides = conn.execute(both_sql, PROBLEM_EDGE_TYPES + CORRECTION_EDGE_TYPES).fetchall()

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "problem_edges": problem_count,
                "correction_edges": correction_count,
                "both_sides_entities": len(both_sides),
                "both_sides_ids": [r[0] for r in both_sides][:10],
                "thesis": "Structural capital concentration creates mechanical both-sides exposure across policy pendulum swings.",
            }
        finally:
            conn.close()

    def get_graph_stats(self) -> dict:
        """Node/edge/claim/source counts."""
        conn = self._conn()
        try:
            stats = {
                "total_nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                "total_edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
                "total_claims": conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0],
                "total_sources": conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0],
            }
            # Top node types
            rows = conn.execute(
                "SELECT node_type, COUNT(*) as cnt FROM nodes GROUP BY node_type ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            stats["top_node_types"] = {r["node_type"]: r["cnt"] for r in rows}
            # Top edge types
            rows = conn.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM edges GROUP BY edge_type ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            stats["top_edge_types"] = {r["edge_type"]: r["cnt"] for r in rows}
            return stats
        finally:
            conn.close()

    def get_both_sides(self, min_confidence: float = 0.7) -> dict:
        """Entities appearing on both problem and correction sides."""
        conn = self._conn()
        try:
            p_ph = ",".join("?" * len(PROBLEM_EDGE_TYPES))
            c_ph = ",".join("?" * len(CORRECTION_EDGE_TYPES))
            sql = """
            SELECT n.node_id, n.name, n.node_type,
                   GROUP_CONCAT(DISTINCT e1.edge_type) as problem_edges,
                   GROUP_CONCAT(DISTINCT e2.edge_type) as correction_edges
            FROM nodes n
            JOIN edges e1 ON n.node_id = e1.from_node_id
            JOIN edges e2 ON n.node_id = e2.from_node_id
            WHERE e1.edge_type IN ({}) AND e2.edge_type IN ({})
            GROUP BY n.node_id
            """.format(p_ph, c_ph)
            rows = conn.execute(sql, PROBLEM_EDGE_TYPES + CORRECTION_EDGE_TYPES).fetchall()
            patterns = [dict(r) for r in rows]
            return {"count": len(patterns), "patterns": patterns}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Tool dispatch (called from local_assistant)
    # ------------------------------------------------------------------

    def call_tool(self, intent: GraphIntent, query: str) -> dict:
        """Dispatch to the right tool based on detected intent."""
        if intent == GraphIntent.SEARCH:
            # Extract search term — strip common prefixes
            term = _extract_search_term(query)
            return self.search_nodes(term)
        elif intent == GraphIntent.EXPLORE:
            node_id = _extract_node_id(query)
            if node_id:
                return self.explore_connections(node_id)
            # Fallback: search first
            term = _extract_search_term(query)
            results = self.search_nodes(term, limit=1)
            if results["count"] > 0:
                return self.explore_connections(results["nodes"][0]["node_id"])
            return {"error": f"No nodes found for: {term}"}
        elif intent == GraphIntent.CAUSAL:
            node_id = _extract_node_id(query)
            return self.find_causal_chains(start_node=node_id)
        elif intent == GraphIntent.THESIS:
            return self.get_thesis_score()
        elif intent == GraphIntent.BOTH_SIDES:
            return self.get_both_sides()
        elif intent == GraphIntent.STATS:
            return self.get_graph_stats()
        else:
            return {"error": f"Unknown intent: {intent}"}

    # ------------------------------------------------------------------
    # Context formatting (for LLM prompt injection)
    # ------------------------------------------------------------------

    def format_for_context(self, result: dict, intent: GraphIntent) -> str:
        """Format graph tool results as concise text for TinyLlama context window."""
        if "error" in result:
            return f"[Graph] Error: {result['error']}"

        if intent == GraphIntent.SEARCH:
            lines = [f"[Graph: {result['count']} nodes found]"]
            for n in result.get("nodes", [])[:8]:
                desc = (n.get("description") or "")[:80]
                lines.append(f"  - {n['node_id']} ({n['node_type']}): {n.get('name', '')} — {desc}")
            return "\n".join(lines)

        elif intent == GraphIntent.EXPLORE:
            node = result.get("node", {})
            lines = [f"[Graph: {node.get('name', node.get('node_id', '?'))} — {result.get('total_connections', 0)} connections]"]
            for c in result.get("connections", [])[:15]:
                e = c["edge"]
                if c["dir"] == "->":
                    lines.append(f"  {c['dir']} {e['edge_type']} -> {e['to_node_id']}")
                else:
                    lines.append(f"  {e['from_node_id']} -> {e['edge_type']} {c['dir']}")
            if result.get("total_connections", 0) > 15:
                lines.append(f"  ... and {result['total_connections'] - 15} more")
            return "\n".join(lines)

        elif intent == GraphIntent.CAUSAL:
            lines = []
            if "chains" in result:
                lines.append(f"[Graph: {result.get('chains_found', 0)} causal chains from {result.get('start', '?')} to {result.get('end', '?')}]")
                for chain in result.get("chains", [])[:5]:
                    lines.append(f"  " + " ".join(str(x) for x in chain))
            elif "paths" in result:
                lines.append(f"[Graph: {result.get('paths_found', 0)} causal paths from {result.get('start', '?')}]")
                for path in result.get("paths", [])[:5]:
                    lines.append(f"  " + " ".join(str(x) for x in path))
            elif "causal_hubs" in result:
                lines.append("[Graph: Top causal hubs]")
                for hub in result["causal_hubs"]:
                    lines.append(f"  - {hub['node_id']}: {hub['outgoing_causal']} outgoing causal edges")
            return "\n".join(lines) if lines else "[Graph] No causal chains found."

        elif intent == GraphIntent.THESIS:
            return (
                f"[Graph: Thesis Score]\n"
                f"  Nodes: {result['total_nodes']}, Edges: {result['total_edges']}\n"
                f"  Problem edges: {result['problem_edges']}, Correction edges: {result['correction_edges']}\n"
                f"  Both-sides entities: {result['both_sides_entities']} {result.get('both_sides_ids', [])}\n"
                f"  Thesis: {result['thesis']}"
            )

        elif intent == GraphIntent.BOTH_SIDES:
            lines = [f"[Graph: {result['count']} both-sides entities]"]
            for p in result.get("patterns", [])[:8]:
                lines.append(
                    f"  - {p.get('name', p.get('node_id', '?'))} ({p.get('node_type', '?')}): "
                    f"problem=[{p.get('problem_edges', '')}] correction=[{p.get('correction_edges', '')}]"
                )
            return "\n".join(lines)

        elif intent == GraphIntent.STATS:
            lines = [
                f"[Graph Stats]",
                f"  Nodes: {result['total_nodes']}, Edges: {result['total_edges']}, "
                f"Claims: {result['total_claims']}, Sources: {result['total_sources']}",
            ]
            if result.get("top_node_types"):
                top = ", ".join(f"{k}:{v}" for k, v in list(result["top_node_types"].items())[:5])
                lines.append(f"  Top node types: {top}")
            if result.get("top_edge_types"):
                top = ", ".join(f"{k}:{v}" for k, v in list(result["top_edge_types"].items())[:5])
                lines.append(f"  Top edge types: {top}")
            return "\n".join(lines)

        return f"[Graph] {json.dumps(result)[:400]}"


# ------------------------------------------------------------------
# Query parsing helpers
# ------------------------------------------------------------------

# Known node IDs / aliases that might appear in queries
_KNOWN_ENTITIES = {
    "blackrock": "blackrock", "vanguard": "vanguard", "intel": "intel",
    "tsmc": "tsmc", "micron": "micron", "nucor": "nucor",
    "chips act": "chips-act", "genius act": "genius-act-2025",
    "sec": "sec", "fdic": "fdic", "fed": "federal-reserve",
    "china": "prc", "russia": "russia", "japan": "japan",
    "heritage foundation": "heritage-foundation",
    "us chamber": "us-chamber-of-commerce",
    "clarence thomas": "clarence-thomas",
    "larry fink": "larry-fink",
}


def _extract_node_id(query: str) -> Optional[str]:
    """Try to extract a node_id from a natural language query."""
    query_lower = query.lower()
    for alias, node_id in _KNOWN_ENTITIES.items():
        if alias in query_lower:
            return node_id
    return None


def _extract_search_term(query: str) -> str:
    """Extract the search target from a query, stripping common prefixes."""
    q = query.strip()
    # Strip common question prefixes
    for prefix in [
        "what do we know about", "tell me about", "who is",
        "what is", "find", "search for", "search graph for",
        "look up", "show me", "graph search",
        "in the graph", "in fgip",
    ]:
        if q.lower().startswith(prefix):
            q = q[len(prefix):].strip()
            break
    # Strip trailing question marks, leading articles, and trailing graph noise
    q = q.rstrip("?").strip()
    for article in ("the ", "a ", "an "):
        if q.lower().startswith(article):
            q = q[len(article):]
    # Strip trailing noise phrases
    for suffix in (" in the graph", " in fgip", " in the database", " from the graph"):
        if q.lower().endswith(suffix):
            q = q[: -len(suffix)]
    return q.strip()
