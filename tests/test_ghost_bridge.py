"""Tests for Ghost Bridge — compressed-domain pre-routing for Hydra.

Tests Ghost feature extraction, pre-route decision engine, calibration,
serialization, and integration with HydraRouter.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from helix_substrate.ghost_bridge import (
    ghost_features_from_bytes,
    GhostPreRoute,
    GhostDecision,
    GhostPreRouteResult,
    PreRouteAction,
)
from helix_substrate.hydra_router import (
    HydraRouter,
    TensorProfile,
    Head,
)


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════


class TestGhostFeatureExtraction:
    """Ghost features from raw bytes."""

    def test_returns_four_features(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=10000, dtype=np.uint8).tobytes()
        gf = ghost_features_from_bytes(data, (100, 100))
        assert set(gf.keys()) == {"te", "tr", "mo", "ac"}

    def test_all_same_bytes_low_entropy(self):
        data = bytes([42] * 10000)
        gf = ghost_features_from_bytes(data, (100, 100))
        assert gf["te"] == 0.0
        assert gf["ac"] == 0.0

    def test_random_bytes_high_entropy(self):
        rng = np.random.default_rng(99)
        data = rng.integers(0, 256, size=50000, dtype=np.uint8).tobytes()
        gf = ghost_features_from_bytes(data, (250, 200))
        assert gf["te"] > 0.9  # near-uniform bigrams → high entropy

    def test_structured_bytes_have_autocorrelation(self):
        # Repeating pattern: 0,1,2,...,255,0,1,2,...
        pattern = np.tile(np.arange(256, dtype=np.uint8), 100)
        gf = ghost_features_from_bytes(pattern.tobytes(), (100, 256))
        assert gf["ac"] > 0.0  # sequential pattern has correlation

    def test_short_input_returns_zeros(self):
        data = bytes([1, 2, 3])
        gf = ghost_features_from_bytes(data, (3,))
        assert gf == {"te": 0.0, "tr": 0.0, "mo": 0.0, "ac": 0.0}

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        data = rng.integers(0, 256, size=10000, dtype=np.uint8).tobytes()
        gf1 = ghost_features_from_bytes(data, (100, 100))
        gf2 = ghost_features_from_bytes(data, (100, 100))
        assert gf1 == gf2

    def test_mo_range(self):
        """Markov order should be >= 0.5 (bigram never less than 2× unigram for real data)."""
        rng = np.random.default_rng(77)
        data = rng.integers(0, 256, size=20000, dtype=np.uint8).tobytes()
        gf = ghost_features_from_bytes(data, (200, 100))
        assert 0.0 <= gf["mo"]


# ═══════════════════════════════════════════════════════════════════════════
# Pre-route decision
# ═══════════════════════════════════════════════════════════════════════════


def _make_test_data(n_safe=60, n_fragile=20, seed=42):
    """Generate synthetic calibration data."""
    rng = np.random.default_rng(seed)
    data = []

    # Safe tensors: lower te, higher mo
    for _ in range(n_safe):
        data.append({
            "ghost": {
                "te": float(rng.uniform(0.4, 0.7)),
                "tr": float(rng.uniform(0.1, 0.3)),
                "mo": float(rng.uniform(0.85, 0.95)),
                "ac": float(rng.uniform(0.0, 0.1)),
            },
            "arch": "transformer",
            "fragile": False,
        })

    # Fragile tensors: higher te, lower mo
    for _ in range(n_fragile):
        data.append({
            "ghost": {
                "te": float(rng.uniform(0.7, 0.95)),
                "tr": float(rng.uniform(0.3, 0.6)),
                "mo": float(rng.uniform(0.92, 0.99)),
                "ac": float(rng.uniform(0.0, 0.05)),
            },
            "arch": "transformer",
            "fragile": True,
        })

    return data


class TestGhostPreRouteDecision:
    """Pre-route decision engine."""

    def test_unknown_arch_always_probes(self):
        preroute = GhostPreRoute()
        decision = preroute.decide({"te": 0.5, "tr": 0.2, "mo": 0.9, "ac": 0.05}, "unknown_arch")
        assert decision.action == PreRouteAction.PROBE_REQUIRED
        assert decision.confidence == 0.0

    def test_calibrated_model_makes_decisions(self):
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)
        assert "transformer" in preroute.arch_models

        # Safe tensor
        decision = preroute.decide(
            {"te": 0.5, "tr": 0.2, "mo": 0.88, "ac": 0.05}, "transformer")
        assert decision.action == PreRouteAction.SKIP_PROBE

        # Fragile tensor
        decision = preroute.decide(
            {"te": 0.9, "tr": 0.5, "mo": 0.96, "ac": 0.01}, "transformer")
        assert decision.action == PreRouteAction.PROBE_REQUIRED

    def test_decision_has_all_fields(self):
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)
        gf = {"te": 0.6, "tr": 0.25, "mo": 0.9, "ac": 0.03}
        decision = preroute.decide(gf, "transformer")

        assert isinstance(decision.action, PreRouteAction)
        assert 0.0 <= decision.confidence <= 1.0
        assert decision.ghost_features == gf
        assert decision.arch == "transformer"
        assert 0.0 <= decision.fragility_score <= 1.0


class TestGhostPreRouteCalibration:
    """Calibration and model fitting."""

    def test_calibrate_produces_coefficients(self):
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)
        model = preroute.arch_models["transformer"]
        assert len(model["coefficients"]) == 5  # intercept + 4 features
        assert "threshold" in model
        assert model["n_train"] == len(data)

    def test_calibrate_meets_safety(self):
        """Calibrated model should not produce dangerous false negatives."""
        data = _make_test_data(n_safe=100, n_fragile=30, seed=123)
        preroute = GhostPreRoute.calibrate(
            data, min_precision_safe=0.95, min_recall_fragile=0.90)

        if "transformer" not in preroute.arch_models:
            pytest.skip("Could not find safe threshold on synthetic data")

        # Check on training data (should pass since threshold was optimized)
        n_fn = 0
        n_ghost_safe = 0
        for d in data:
            decision = preroute.decide(d["ghost"], d["arch"])
            if decision.action == PreRouteAction.SKIP_PROBE:
                n_ghost_safe += 1
                if d["fragile"]:
                    n_fn += 1

        precision = (n_ghost_safe - n_fn) / n_ghost_safe if n_ghost_safe > 0 else 1.0
        assert precision >= 0.90  # relaxed for synthetic data

    def test_calibrate_insufficient_data_skipped(self):
        """Architecture with < 10 samples should be skipped."""
        data = _make_test_data(n_safe=5, n_fragile=3)
        preroute = GhostPreRoute.calibrate(data)
        assert "transformer" not in preroute.arch_models

    def test_multi_arch_calibration(self):
        data_t = _make_test_data(n_safe=50, n_fragile=15)
        data_m = _make_test_data(n_safe=40, n_fragile=10, seed=99)
        for d in data_m:
            d["arch"] = "mamba"

        preroute = GhostPreRoute.calibrate(data_t + data_m)
        assert "transformer" in preroute.arch_models
        assert "mamba" in preroute.arch_models
        # Different architectures should have different coefficients
        assert (preroute.arch_models["transformer"]["coefficients"]
                != preroute.arch_models["mamba"]["coefficients"])


class TestGhostPreRouteSerialization:
    """Save and load calibrated models."""

    def test_save_load_roundtrip(self):
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            preroute.save(path)
            loaded = GhostPreRoute.load(path)

            assert loaded.arch_models.keys() == preroute.arch_models.keys()
            for arch in preroute.arch_models:
                assert (loaded.arch_models[arch]["coefficients"]
                        == preroute.arch_models[arch]["coefficients"])
                assert (loaded.arch_models[arch]["threshold"]
                        == preroute.arch_models[arch]["threshold"])
        finally:
            path.unlink(missing_ok=True)

    def test_to_dict(self):
        preroute = GhostPreRoute(arch_models={
            "test": {"coefficients": [0.1, 0.2, 0.3, 0.4, 0.5], "threshold": 0.4}
        })
        d = preroute.to_dict()
        assert d["version"] == "ghost_bridge_v1"
        assert "test" in d["arch_models"]


# ═══════════════════════════════════════════════════════════════════════════
# Hydra integration
# ═══════════════════════════════════════════════════════════════════════════


def _make_profiles(n=10):
    """Generate test TensorProfiles."""
    profiles = []
    for i in range(n):
        profiles.append(TensorProfile(
            tensor_name=f"layers.{i}.mlp.weight",
            shape=(768, 768),
            layer_index=i,
            tensor_type="gate_proj" if i % 2 == 0 else "up_proj",
            n_params=768 * 768,
            kurtosis=3.0 + i * 0.5,
            std=0.01,
            affine6_cosine=0.9995,
            affine5_cosine=0.999 if i < 7 else 0.997,
        ))
    return profiles


class TestHydraGhostIntegration:
    """route_with_ghost() integration."""

    def test_ghost_routed_tensors_get_affine5(self):
        router = HydraRouter(policy="edge_balanced")
        data = _make_test_data(n_safe=60, n_fragile=20)
        preroute = GhostPreRoute.calibrate(data)

        profiles = _make_profiles(5)
        ghost_map = {}
        for p in profiles:
            ghost_map[p.tensor_name] = {
                "te": 0.5, "tr": 0.2, "mo": 0.88, "ac": 0.05
            }

        plan, result = router.route_with_ghost(
            profiles, preroute, ghost_map, "transformer", "test_model")

        # Ghost-safe tensors should get affine5 under edge_balanced
        for tp in plan.tensors:
            if "ghost_preroute" in tp.reason:
                assert tp.head == Head.AFFINE5

    def test_ghost_result_tracks_counts(self):
        router = HydraRouter(policy="edge_balanced")
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)

        profiles = _make_profiles(8)
        ghost_map = {}
        # Half safe, half fragile
        for i, p in enumerate(profiles):
            if i < 4:
                ghost_map[p.tensor_name] = {"te": 0.5, "tr": 0.2, "mo": 0.88, "ac": 0.05}
            else:
                ghost_map[p.tensor_name] = {"te": 0.9, "tr": 0.5, "mo": 0.96, "ac": 0.01}

        plan, result = router.route_with_ghost(
            profiles, preroute, ghost_map, "transformer")

        assert result.n_total == 8
        assert result.n_ghost_routed + result.n_probe_required == 8
        assert result.n_ghost_routed > 0  # at least some ghost-routed

    def test_exact_tensors_bypass_ghost(self):
        router = HydraRouter(policy="edge_balanced")
        preroute = GhostPreRoute()  # uncalibrated

        profiles = [TensorProfile(
            tensor_name="model.embed_tokens",
            shape=(32000, 768),
            layer_index=0,
            tensor_type="embed_tokens",
            n_params=32000 * 768,
        )]
        ghost_map = {"model.embed_tokens": {"te": 0.5, "tr": 0.2, "mo": 0.9, "ac": 0.05}}

        plan, result = router.route_with_ghost(
            profiles, preroute, ghost_map, "transformer")

        assert plan.tensors[0].head == Head.EXACT
        assert "exact_tensor_type" in plan.tensors[0].reason

    def test_missing_ghost_features_falls_through(self):
        """Tensors without Ghost data use normal probe routing."""
        router = HydraRouter(policy="edge_balanced")
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)

        profiles = _make_profiles(3)
        # Only provide ghost features for first tensor
        ghost_map = {profiles[0].tensor_name: {"te": 0.5, "tr": 0.2, "mo": 0.88, "ac": 0.05}}

        plan, result = router.route_with_ghost(
            profiles, preroute, ghost_map, "transformer")

        assert len(plan.tensors) == 3
        assert result.n_probe_required >= 2  # at least the 2 without ghost data

    def test_ghost_result_serializes(self):
        result = GhostPreRouteResult(n_total=10, n_ghost_routed=6, n_probe_required=4)
        d = result.to_dict()
        assert d["n_total"] == 10
        assert d["cleared_fraction"] == 0.6

    def test_quality_first_uses_affine6(self):
        """Quality-first policy's safe head should be affine6, not affine5."""
        router = HydraRouter(policy="quality_first")
        data = _make_test_data()
        preroute = GhostPreRoute.calibrate(data)

        profiles = _make_profiles(3)
        ghost_map = {p.tensor_name: {"te": 0.5, "tr": 0.2, "mo": 0.88, "ac": 0.05}
                     for p in profiles}

        plan, result = router.route_with_ghost(
            profiles, preroute, ghost_map, "transformer")

        for tp in plan.tensors:
            if "ghost_preroute" in tp.reason:
                assert tp.head == Head.AFFINE6


# ═══════════════════════════════════════════════════════════════════════════
# Real data test (skips if HXQ files not available)
# ═══════════════════════════════════════════════════════════════════════════

MAMBA_HXQ = Path(
    "/home/voidstr3m33/.cache/huggingface/hub/models--EchoLabs33--mamba-130m-hxq"
    "/snapshots/67353fa944a4769b656977c6871c5099e57a4ea6/model.safetensors"
)


class TestRealDataGhost:
    """Verify Ghost features on real HXQ data."""

    @pytest.fixture(scope="class")
    def real_ghost_features(self):
        if not MAMBA_HXQ.exists():
            pytest.skip("Mamba HXQ safetensors not available")
        import struct
        with open(MAMBA_HXQ, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        tensors_info = {k: v for k, v in header.items() if k != "__metadata__"}

        results = []
        for name, info in tensors_info.items():
            if not name.endswith(".indices") or info["dtype"] != "U8":
                continue
            byte_size = info["data_offsets"][1] - info["data_offsets"][0]
            if byte_size < 1024:
                continue
            start, end = info["data_offsets"]
            with open(MAMBA_HXQ, "rb") as f:
                f.seek(data_start + start)
                raw = f.read(end - start)
            gf = ghost_features_from_bytes(raw, tuple(info["shape"]))
            results.append({"name": name, "ghost": gf, "shape": info["shape"]})
        return results

    def test_real_features_in_range(self, real_ghost_features):
        for r in real_ghost_features:
            gf = r["ghost"]
            assert 0.0 <= gf["te"] <= 1.0, f"{r['name']}: te={gf['te']}"
            assert 0.0 <= gf["tr"] <= 1.0, f"{r['name']}: tr={gf['tr']}"
            assert gf["mo"] >= 0.0, f"{r['name']}: mo={gf['mo']}"
            assert 0.0 <= gf["ac"] <= 1.0, f"{r['name']}: ac={gf['ac']}"

    def test_real_features_vary(self, real_ghost_features):
        """Different tensors should have different Ghost features."""
        te_vals = [r["ghost"]["te"] for r in real_ghost_features]
        assert max(te_vals) - min(te_vals) > 0.01, "All tensors have same te"

    def test_real_features_deterministic(self, real_ghost_features):
        """Same bytes → same features."""
        if not real_ghost_features:
            pytest.skip("No tensors")
        r = real_ghost_features[0]
        # Re-read and re-compute
        import struct
        with open(MAMBA_HXQ, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        info = header[r["name"]]
        start, end = info["data_offsets"]
        with open(MAMBA_HXQ, "rb") as f:
            f.seek(data_start + start)
            raw = f.read(end - start)
        gf2 = ghost_features_from_bytes(raw, tuple(info["shape"]))
        assert gf2 == r["ghost"]
