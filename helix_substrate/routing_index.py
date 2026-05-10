"""
routing_index.py ��� Layer 3: Compression routing index.

SQLite-backed index over routing context data from two sources:
  1. HXZO v2 sidecar headers (routing_context block, 17 fields per tensor)
  2. Recompression event JSONL logs (per-step, per-layer measurements)

Provides a query interface for the codec to answer questions like:
  - "What do SSM in_proj layers look like across all architectures?"
  - "Which layers have eff_rank < 0.5 at step 200?"
  - "What's the mean drift_ratio for attention layers vs FFN layers?"

Usage:
    from helix_substrate.routing_index import RoutingIndex

    idx = RoutingIndex("routing_index.db")
    idx.ingest_events_jsonl("receipts/echo_hybrid/wo_echo_hybrid_04_recomp_events.jsonl")
    idx.ingest_hxzo_dir("/path/to/sidecars/")

    # Query
    rows = idx.query_events(block_type="ssm", role="in_proj", step_min=100)
    stats = idx.aggregate_routing_context(tensor_class="SSM", arch="zamba2")
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Schema ──

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS routing_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    layer_name TEXT NOT NULL,
    block_type TEXT,
    role TEXT,
    shape TEXT,
    n_params INTEGER,
    eff_rank REAL,
    se REAL,
    kurtosis REAL,
    weight_rms REAL,
    weight_std REAL,
    pre_sidecar_norm REAL,
    post_sidecar_norm REAL,
    drift_ratio REAL,
    threshold_used REAL,
    loss_at_recomp REAL,
    init_eff_rank REAL,
    eff_rank_delta REAL,
    init_kurtosis REAL,
    kurtosis_delta REAL,
    config TEXT,
    source_file TEXT,
    UNIQUE(step, layer_name, config)
);
"""

_CONTEXT_DDL = """
CREATE TABLE IF NOT EXISTS routing_context (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tensor_name TEXT NOT NULL,
    tensor_class TEXT,
    block_type TEXT,
    role TEXT,
    block_idx INTEGER,
    arch TEXT,
    eff_rank REAL,
    kurtosis REAL,
    se REAL,
    weight_rms REAL,
    route TEXT,
    composite_score REAL,
    confidence REAL,
    pre_sidecar_norm REAL,
    post_sidecar_norm REAL,
    compression_step INTEGER,
    drift_ratio REAL,
    recomp_count INTEGER,
    source_file TEXT,
    UNIQUE(tensor_name, source_file)
);
"""

_INDEXES_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_events_step ON routing_events(step);",
    "CREATE INDEX IF NOT EXISTS idx_events_block_role ON routing_events(block_type, role);",
    "CREATE INDEX IF NOT EXISTS idx_events_layer ON routing_events(layer_name);",
    "CREATE INDEX IF NOT EXISTS idx_context_class ON routing_context(tensor_class);",
    "CREATE INDEX IF NOT EXISTS idx_context_arch ON routing_context(arch);",
    "CREATE INDEX IF NOT EXISTS idx_context_block_role ON routing_context(block_type, role);",
]


class RoutingIndex:
    """SQLite-backed routing index for compression decisions."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        cur.executescript(_EVENTS_DDL + _CONTEXT_DDL)
        for idx_sql in _INDEXES_DDL:
            cur.execute(idx_sql)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # ── Ingest ──

    def ingest_events_jsonl(self, jsonl_path: str) -> int:
        """Load recompression events from a JSONL file. Returns count ingested."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(jsonl_path)

        source = str(path.name)
        count = 0
        cur = self.conn.cursor()

        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            try:
                cur.execute(
                    """INSERT OR IGNORE INTO routing_events
                    (step, layer_name, block_type, role, shape, n_params,
                     eff_rank, se, kurtosis, weight_rms, weight_std,
                     pre_sidecar_norm, post_sidecar_norm, drift_ratio,
                     threshold_used, loss_at_recomp, init_eff_rank,
                     eff_rank_delta, init_kurtosis, kurtosis_delta,
                     config, source_file)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        evt.get("step"),
                        evt.get("layer_name"),
                        evt.get("block_type"),
                        evt.get("role"),
                        json.dumps(evt.get("shape")) if evt.get("shape") else None,
                        evt.get("n_params"),
                        evt.get("eff_rank"),
                        evt.get("se"),
                        evt.get("kurtosis"),
                        evt.get("weight_rms"),
                        evt.get("weight_std"),
                        evt.get("pre_sidecar_norm"),
                        evt.get("post_sidecar_norm"),
                        evt.get("drift_ratio"),
                        evt.get("threshold_used"),
                        evt.get("loss_at_recomp"),
                        evt.get("init_eff_rank"),
                        evt.get("eff_rank_delta"),
                        evt.get("init_kurtosis"),
                        evt.get("kurtosis_delta"),
                        evt.get("config"),
                        source,
                    ),
                )
                count += 1
            except sqlite3.IntegrityError:
                pass

        self.conn.commit()
        return count

    def ingest_hxzo_header(self, routing_context: Dict[str, Any],
                           tensor_name: str, source_file: str = "inline") -> bool:
        """Ingest a single routing_context dict (from HXZO v2 header or live)."""
        rc = routing_context
        cur = self.conn.cursor()
        try:
            cur.execute(
                """INSERT OR REPLACE INTO routing_context
                (tensor_name, tensor_class, block_type, role, block_idx, arch,
                 eff_rank, kurtosis, se, weight_rms, route, composite_score,
                 confidence, pre_sidecar_norm, post_sidecar_norm,
                 compression_step, drift_ratio, recomp_count, source_file)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    tensor_name,
                    rc.get("tensor_class"),
                    rc.get("block_type"),
                    rc.get("role"),
                    rc.get("block_idx"),
                    rc.get("arch"),
                    rc.get("eff_rank"),
                    rc.get("kurtosis"),
                    rc.get("se"),
                    rc.get("weight_rms"),
                    rc.get("route"),
                    rc.get("composite_score"),
                    rc.get("confidence"),
                    rc.get("pre_sidecar_norm"),
                    rc.get("post_sidecar_norm"),
                    rc.get("compression_step"),
                    rc.get("drift_ratio"),
                    rc.get("recomp_count"),
                    source_file,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.Error:
            return False

    def ingest_hxzo_dir(self, sidecar_dir: str) -> int:
        """Scan a directory for .hxzo files and ingest routing_context from v2 headers."""
        from helix_substrate.sidecar import inspect_hxzo_header

        count = 0
        for hxzo_path in Path(sidecar_dir).rglob("*.hxzo"):
            try:
                hdr = inspect_hxzo_header(str(hxzo_path))
                if not hdr.get("valid"):
                    continue
                rc = hdr.get("routing_context")
                if rc is None:
                    continue
                tensor_name = hdr.get("tensor_name", str(hxzo_path.stem))
                if self.ingest_hxzo_header(rc, tensor_name, str(hxzo_path.name)):
                    count += 1
            except Exception:
                continue
        return count

    # ── Query: Events ──

    def query_events(
        self,
        block_type: Optional[str] = None,
        role: Optional[str] = None,
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
        layer_name: Optional[str] = None,
        config: Optional[str] = None,
        eff_rank_max: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query routing events with optional filters."""
        clauses = []
        params = []

        if block_type:
            clauses.append("block_type = ?")
            params.append(block_type)
        if role:
            clauses.append("role = ?")
            params.append(role)
        if step_min is not None:
            clauses.append("step >= ?")
            params.append(step_min)
        if step_max is not None:
            clauses.append("step <= ?")
            params.append(step_max)
        if layer_name:
            clauses.append("layer_name LIKE ?")
            params.append(f"%{layer_name}%")
        if config:
            clauses.append("config = ?")
            params.append(config)
        if eff_rank_max is not None:
            clauses.append("eff_rank <= ?")
            params.append(eff_rank_max)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM routing_events WHERE {where} ORDER BY step, layer_name LIMIT ?"
        params.append(limit)

        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    # ── Query: Routing Context ──

    def query_context(
        self,
        tensor_class: Optional[str] = None,
        block_type: Optional[str] = None,
        role: Optional[str] = None,
        arch: Optional[str] = None,
        route: Optional[str] = None,
        eff_rank_max: Optional[float] = None,
        confidence_min: Optional[float] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query routing context records with optional filters."""
        clauses = []
        params = []

        if tensor_class:
            clauses.append("tensor_class = ?")
            params.append(tensor_class)
        if block_type:
            clauses.append("block_type = ?")
            params.append(block_type)
        if role:
            clauses.append("role = ?")
            params.append(role)
        if arch:
            clauses.append("arch = ?")
            params.append(arch)
        if route:
            clauses.append("route = ?")
            params.append(route)
        if eff_rank_max is not None:
            clauses.append("eff_rank <= ?")
            params.append(eff_rank_max)
        if confidence_min is not None:
            clauses.append("confidence >= ?")
            params.append(confidence_min)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM routing_context WHERE {where} ORDER BY block_idx, tensor_name LIMIT ?"
        params.append(limit)

        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    # ── Aggregation ──

    def aggregate_events(
        self,
        group_by: str = "block_type",
        step_min: Optional[int] = None,
        step_max: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate event statistics grouped by a column."""
        valid_groups = {"block_type", "role", "config", "layer_name"}
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        clauses = []
        params = []
        if step_min is not None:
            clauses.append("step >= ?")
            params.append(step_min)
        if step_max is not None:
            clauses.append("step <= ?")
            params.append(step_max)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"""
            SELECT {group_by},
                   COUNT(*) as n,
                   AVG(eff_rank) as mean_eff_rank,
                   AVG(kurtosis) as mean_kurtosis,
                   AVG(se) as mean_se,
                   AVG(drift_ratio) as mean_drift_ratio,
                   AVG(pre_sidecar_norm) as mean_pre_sidecar_norm,
                   AVG(post_sidecar_norm) as mean_post_sidecar_norm,
                   AVG(loss_at_recomp) as mean_loss_at_recomp,
                   MIN(eff_rank) as min_eff_rank,
                   MAX(eff_rank) as max_eff_rank
            FROM routing_events
            WHERE {where}
            GROUP BY {group_by}
            ORDER BY {group_by}
        """
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def aggregate_context(
        self,
        group_by: str = "tensor_class",
        arch: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Aggregate routing context statistics grouped by a column."""
        valid_groups = {"tensor_class", "block_type", "role", "arch", "route"}
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        clauses = []
        params = []
        if arch:
            clauses.append("arch = ?")
            params.append(arch)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"""
            SELECT {group_by},
                   COUNT(*) as n,
                   AVG(eff_rank) as mean_eff_rank,
                   AVG(kurtosis) as mean_kurtosis,
                   AVG(se) as mean_se,
                   AVG(confidence) as mean_confidence,
                   AVG(pre_sidecar_norm) as mean_pre_norm,
                   AVG(post_sidecar_norm) as mean_post_norm,
                   MIN(eff_rank) as min_eff_rank,
                   MAX(eff_rank) as max_eff_rank
            FROM routing_context
            WHERE {where}
            GROUP BY {group_by}
            ORDER BY {group_by}
        """
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    # ── Drift analysis ──

    def layer_drift_over_steps(self, layer_name: str) -> List[Dict[str, Any]]:
        """Get eff_rank and kurtosis evolution for a specific layer over training steps."""
        cur = self.conn.execute(
            """SELECT step, eff_rank, kurtosis, se, drift_ratio,
                      pre_sidecar_norm, loss_at_recomp, eff_rank_delta
               FROM routing_events
               WHERE layer_name = ?
               ORDER BY step""",
            (layer_name,),
        )
        return [dict(row) for row in cur.fetchall()]

    # ── Stats ──

    def stats(self) -> Dict[str, int]:
        """Return row counts for both tables."""
        evt_count = self.conn.execute("SELECT COUNT(*) FROM routing_events").fetchone()[0]
        ctx_count = self.conn.execute("SELECT COUNT(*) FROM routing_context").fetchone()[0]
        return {"routing_events": evt_count, "routing_context": ctx_count}
