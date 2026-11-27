"""Smoke tests. Need at least one .mpk package in models/ (skipped otherwise).   Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cooling_ai import CoolingEngine, install_mpk   # noqa: E402

PKGS = sorted((ROOT / "models").glob("*.mpk"))
pytestmark = pytest.mark.skipif(not PKGS, reason="no .mpk package in models/")


@pytest.fixture(scope="module")
def engine():
    return CoolingEngine.from_mpk(PKGS[-1])


@pytest.fixture(scope="module")
def replay(engine):
    df = engine.demo_data()
    assert df is not None, "package has no bundled replay"
    return df


def test_predictions_are_sane(engine, replay):
    res = engine.run(replay)
    ok = res.dropna(subset=["setpoint_ai"])
    lo, hi = engine.meta["setpoint_range_C"]
    assert len(ok) > 0
    assert ok["setpoint_ai"].between(lo, hi).all()
    assert ok["hot_prob"].between(0, 1).all()
    assert ok["cpu_pred"].between(-0.2, 1.2).all()


def test_decide_matches_batch_run(engine, replay):
    res = engine.run(replay)
    d = engine.decide(replay)
    assert d.timestamp == res.index[-1]
    assert d.setpoint_c == pytest.approx(res["setpoint_ai"].iloc[-1], abs=1e-4)


def test_safety_layer_caps_setpoint(engine):
    feats = {"cpu_now": 0.9, "cpu_pred": 0.9, "hot_prob": 0.99, "temp_mean": 24.0, "temp_max": 30.0,
             "hum": 40.0, "hour_sin": 0.0, "hour_cos": 1.0}
    sp, rf_sp, override = engine.decide_from_features(feats)
    assert override and sp <= engine.meta["base_setpoint_C"] - 2.0 + 1e-9


def test_energy_report(engine, replay):
    summary, log = engine.energy_report(engine.run(replay))
    assert summary["cooling_kwh_baseline"] > 0
    assert summary["pue_ai"] >= 1 and summary["pue_baseline"] >= 1
    assert np.isfinite(log["cooling_kw_ai"]).all()


def test_rejects_garbage(tmp_path):
    with pytest.raises(ValueError):
        install_mpk(b"not a package", tmp_path)
