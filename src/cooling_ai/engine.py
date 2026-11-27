"""cooling_ai.engine - load a ``.mpk`` model package and run the AI cooling controller.

Pipeline (identical to the training notebook):

    sensors + CPU load  ->  LSTM  -> CPU forecast (next HORIZON steps)
    sensor thermal maps ->  CNN   -> probability of a hotspot
    [forecast, hotspot prob, temps, humidity, hour]  ->  Random Forest  -> HVAC supply-air setpoint
                                                         + safety layer (cool harder if hotspot risk is high)
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")      # keep TensorFlow quiet

MPK_FORMAT = "datacenter-cooling-models"
MODEL_FILES = ("lstm_cpu_forecast.keras", "cnn_hotspot.keras", "rf_hvac_controller.joblib")
REPLAY_FILE = "demo_replay.csv"

PathOrBytes = Union[str, os.PathLike, bytes, bytearray]


# --------------------------------------------------------------------------- helpers
def cop(setpoint_c):
    """CRAC coefficient of performance as a function of the supply-air temperature (deg C)."""
    t = np.asarray(setpoint_c, dtype=float)
    return 0.0068 * t ** 2 + 0.0008 * t + 0.458


def install_mpk(source: PathOrBytes, dest: Union[str, os.PathLike]) -> dict:
    """Unpack a ``.mpk`` package into ``dest`` and return its metadata dict."""
    import msgpack

    raw = bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    try:
        bundle = msgpack.unpackb(raw, raw=False)
    except Exception as exc:                                     # corrupted / not msgpack
        raise ValueError(f"Not a valid .mpk file: {exc}") from exc
    if not isinstance(bundle, dict) or bundle.get("format") != MPK_FORMAT:
        raise ValueError(f"Not a '{MPK_FORMAT}' package")
    missing = [f for f in MODEL_FILES if f not in bundle.get("files", {})]
    if missing:
        raise ValueError(f"Package is missing model files: {missing}")

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name, blob in bundle["files"].items():
        (dest / Path(name).name).write_bytes(blob)               # .name = no path traversal
    (dest / "meta.json").write_text(json.dumps(bundle["meta"], indent=2))
    return bundle["meta"]


def load_csv(source, meta: dict | None = None) -> pd.DataFrame:
    """Read a sensor/workload CSV into the layout the engine expects.

    Required columns: a time column (``time`` / ``timestamp`` / ``date``...), ``cpu`` (0-1 or 0-100 %),
    the sensor columns ``T_0 ... T_{n-1}``; optional ``hum``. The data is resampled to the model's time step.
    """
    df = pd.read_csv(source)
    tcol = next((c for c in df.columns if any(k in c.lower() for k in ("time", "date", "stamp"))), df.columns[0])
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
    df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
    df.index.name = "time"
    df = df.apply(pd.to_numeric, errors="coerce")
    if "cpu" in df and df["cpu"].max() > 1.5:                    # percent -> fraction
        df["cpu"] = df["cpu"] / 100.0
    if "cpu" in df:
        df["cpu"] = df["cpu"].clip(0, 1)
    if meta is not None:
        df = df.resample(meta["step"]).mean().interpolate(limit_direction="both")
    return df


@dataclass
class Decision:
    """Result of one control decision."""
    timestamp: pd.Timestamp
    setpoint_c: float            # final HVAC supply-air setpoint (after the safety layer)
    rf_setpoint_c: float         # raw Random Forest output
    baseline_setpoint_c: float   # fixed-setpoint policy we compare to
    hot_prob: float              # CNN probability of a hotspot in the next HORIZON steps
    cpu_now: float
    cpu_forecast: np.ndarray     # LSTM forecast, shape (HORIZON,)
    safety_override: bool        # True if the hotspot-risk safety layer lowered the setpoint

    def as_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "setpoint_c": round(self.setpoint_c, 2),
            "rf_setpoint_c": round(self.rf_setpoint_c, 2),
            "baseline_setpoint_c": self.baseline_setpoint_c,
            "hotspot_probability": round(self.hot_prob, 4),
            "cpu_now": round(self.cpu_now, 4),
            "cpu_forecast": [round(float(v), 4) for v in self.cpu_forecast],
            "safety_override": self.safety_override,
        }


# --------------------------------------------------------------------------- engine
class CoolingEngine:
    """Holds the three models + metadata and exposes ``run`` / ``decide`` / ``energy_report``."""

    def __init__(self, meta: dict, lstm, cnn, rf, model_dir: Path):
        self.meta, self.lstm, self.cnn, self.rf, self.model_dir = meta, lstm, cnn, rf, Path(model_dir)
        self._dmin = np.asarray(meta["scaler_data_min"], dtype="float32")
        self._drng = np.asarray(meta["scaler_data_range"], dtype="float32")
        self._cpu_idx = meta["lstm_features"].index("cpu")

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_mpk(cls, source: PathOrBytes, cache_dir: Union[str, os.PathLike, None] = None) -> "CoolingEngine":
        """Install a ``.mpk`` package (path or raw bytes) and load the models."""
        import joblib
        import tensorflow as tf

        model_dir = Path(cache_dir) if cache_dir else Path(tempfile.mkdtemp(prefix="cooling_mpk_"))
        meta = install_mpk(source, model_dir)
        lstm = tf.keras.models.load_model(model_dir / MODEL_FILES[0], compile=False)
        cnn = tf.keras.models.load_model(model_dir / MODEL_FILES[1], compile=False)
        rf = joblib.load(model_dir / MODEL_FILES[2])
        return cls(meta, lstm, cnn, rf, model_dir)

    # ---- data ---------------------------------------------------------------
    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic_data", False))

    def demo_data(self) -> pd.DataFrame | None:
        """Replay sample bundled inside the package (7 days of the test period)."""
        p = self.model_dir / REPLAY_FILE
        return load_csv(p, self.meta) if p.exists() else None

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the input and add the derived columns used by the models."""
        m = self.meta
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex (use load_csv).")
        need = list(m["sensor_cols"]) + ["cpu"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}. "
                f"Expected: time index, 'cpu' (0-1), {m['n_sensors']} sensor columns "
                f"{m['sensor_cols'][0]}..{m['sensor_cols'][-1]} and optionally 'hum'.")
        d = df.sort_index().copy()
        if "hum" not in d:
            d["hum"] = 45.0
        cols = list(m["sensor_cols"]) + ["cpu", "hum"]
        d[cols] = d[cols].interpolate(limit_direction="both")
        d["temp_mean"] = d[m["sensor_cols"]].mean(axis=1)
        d["temp_max"] = d[m["sensor_cols"]].max(axis=1)
        hod = d.index.hour + d.index.minute / 60
        d["hour_sin"] = np.sin(2 * np.pi * hod / 24)
        d["hour_cos"] = np.cos(2 * np.pi * hod / 24)
        return d[cols + ["temp_mean", "temp_max", "hour_sin", "hour_cos"]]

    def to_grid(self, values: np.ndarray) -> np.ndarray:
        """(t, n_sensors) -> (t, G, G) thermal maps; unused cells are padded with the row mean."""
        v = np.atleast_2d(np.asarray(values, dtype="float32"))
        g = self.meta["grid_side"]
        pad = g * g - v.shape[1]
        if pad:
            v = np.hstack([v, np.repeat(v.mean(axis=1, keepdims=True), pad, axis=1)])
        return v.reshape(-1, g, g)

    # ---- models -------------------------------------------------------------
    def _forecast(self, d: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        """LSTM CPU forecast for every time step that has enough history -> (N, HORIZON), NaN elsewhere."""
        L, H = self.meta["lookback"], self.meta["horizon"]
        N = len(d)
        out = np.full((N, H), np.nan, dtype="float32")
        if N < L:
            return out
        x = (d[self.meta["lstm_features"]].to_numpy("float32") - self._dmin) / self._drng
        win = np.lib.stride_tricks.sliding_window_view(x, L, axis=0)        # (N-L+1, F, L)
        win = np.ascontiguousarray(win.transpose(0, 2, 1))                  # (N-L+1, L, F)
        pred = self.lstm.predict(win, batch_size=batch_size, verbose=0)
        out[L - 1:] = pred * self._drng[self._cpu_idx] + self._dmin[self._cpu_idx]
        return out

    def _hot_prob(self, d: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
        """CNN hotspot probability for every step that has enough history -> (N,), NaN elsewhere."""
        m = self.meta
        lags = list(m["cnn_lags"])
        tmin, N = max(lags), len(d)
        out = np.full(N, np.nan, dtype="float32")
        if N <= tmin:
            return out
        grids = (self.to_grid(d[m["sensor_cols"]].to_numpy()) - m["cnn_norm_mean"]) / m["cnn_norm_std"]
        xc = np.stack([grids[tmin - l: N - l] for l in lags], axis=-1).astype("float32")
        out[tmin:] = self.cnn.predict(xc, batch_size=batch_size, verbose=0).ravel()
        return out

    def _apply_safety(self, rf_sp: np.ndarray, hot_prob: np.ndarray):
        m = self.meta
        override = hot_prob > m["hot_prob_safety_limit"]
        sp = np.where(override, np.minimum(rf_sp, m["base_setpoint_C"] - 2.0), rf_sp)
        lo, hi = m["setpoint_range_C"]
        return np.clip(sp, lo, hi), override

    # ---- public API ---------------------------------------------------------
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run LSTM + CNN + RF over the whole series.

        Returns one row per time step with the model inputs/outputs. Rows without enough history
        (the first ``lookback`` steps) have NaN predictions and are skipped by ``energy_report``.
        """
        m = self.meta
        d = self.prepare(df)
        fc = self._forecast(d)
        hp = self._hot_prob(d)
        res = pd.DataFrame({
            "cpu_now": d["cpu"], "cpu_pred": fc.mean(axis=1), "hot_prob": hp,
            "temp_mean": d["temp_mean"], "temp_max": d["temp_max"], "hum": d["hum"],
            "hour_sin": d["hour_sin"], "hour_cos": d["hour_cos"],
        }, index=d.index)
        res["rf_setpoint"] = np.nan
        res["setpoint_ai"] = np.nan
        res["safety_override"] = False
        ok = res[["cpu_pred", "hot_prob"]].notna().all(axis=1).to_numpy()
        if ok.any():
            rf_sp = self.rf.predict(res.loc[ok, m["rf_features"]])
            sp, override = self._apply_safety(rf_sp, res.loc[ok, "hot_prob"].to_numpy())
            res.loc[ok, "rf_setpoint"] = rf_sp
            res.loc[ok, "setpoint_ai"] = sp
            res.loc[ok, "safety_override"] = override
        for k in range(m["horizon"]):
            res[f"cpu_f{k + 1}"] = fc[:, k]
        return res

    def decide(self, df: pd.DataFrame) -> Decision:
        """Control decision for the LAST time step of ``df`` (needs >= ``lookback`` rows of history)."""
        L = self.meta["lookback"]
        if len(df) < L:
            raise ValueError(f"Need at least {L} time steps of history, got {len(df)}.")
        res = self.run(df.sort_index().tail(L))
        return self._decision_from_row(res.iloc[-1], res.index[-1])

    def _decision_from_row(self, row: pd.Series, ts) -> Decision:
        H = self.meta["horizon"]
        return Decision(
            timestamp=pd.Timestamp(ts), setpoint_c=float(row["setpoint_ai"]), rf_setpoint_c=float(row["rf_setpoint"]),
            baseline_setpoint_c=float(self.meta["base_setpoint_C"]), hot_prob=float(row["hot_prob"]),
            cpu_now=float(row["cpu_now"]), cpu_forecast=np.array([row[f"cpu_f{k + 1}"] for k in range(H)], dtype=float),
            safety_override=bool(row["safety_override"]),
        )

    def decide_from_features(self, features: dict) -> tuple[float, float, bool]:
        """What-if: Random Forest + safety layer on hand-made features -> (setpoint, raw RF setpoint, override)."""
        row = pd.DataFrame([features])[self.meta["rf_features"]]
        rf_sp = self.rf.predict(row)
        sp, override = self._apply_safety(rf_sp, np.array([float(features["hot_prob"])]))
        return float(sp[0]), float(rf_sp[0]), bool(override[0])

    def energy_report(self, res: pd.DataFrame, params: dict | None = None) -> tuple[dict, pd.DataFrame]:
        """Energy / cost / CO2 / PUE of the AI controller vs the fixed-setpoint baseline.

        ``params`` can override the energy assumptions stored in the package
        (n_servers, p_idle_kw, p_max_kw, k_pass, co2_kg_per_kwh, price_per_kwh, base_setpoint_C).
        Returns (summary dict, per-step DataFrame).
        """
        m = self.meta
        p = {**m["energy"], "base_setpoint_C": m["base_setpoint_C"], **(params or {})}
        R = res[res["setpoint_ai"].notna()].copy()
        if R.empty:
            raise ValueError("No time step with a valid prediction (need more history).")
        dt_h = pd.Timedelta(m["step"]).total_seconds() / 3600
        base_sp, sp = p["base_setpoint_C"], R["setpoint_ai"].to_numpy()
        R["it_kw"] = p["n_servers"] * (p["p_idle_kw"] + (p["p_max_kw"] - p["p_idle_kw"]) * R["cpu_now"])
        R["cooling_kw_baseline"] = R["it_kw"] / cop(base_sp)
        R["cooling_kw_ai"] = R["it_kw"] / cop(sp)
        R["temp_max_ai"] = R["temp_max"] + p["k_pass"] * (sp - base_sp)
        kwh_b, kwh_a = R["cooling_kw_baseline"].sum() * dt_h, R["cooling_kw_ai"].sum() * dt_h
        saved = kwh_b - kwh_a
        thr = m["hot_threshold_C"]
        summary = {
            "steps": int(len(R)), "days": round(len(R) * dt_h / 24, 2),
            "cooling_kwh_baseline": round(float(kwh_b), 1), "cooling_kwh_ai": round(float(kwh_a), 1),
            "energy_saved_kwh": round(float(saved), 1),
            "energy_saved_pct": round(float(100 * saved / kwh_b), 2) if kwh_b else 0.0,
            "co2_avoided_kg": round(float(saved * p["co2_kg_per_kwh"]), 1),
            "cost_saved": round(float(saved * p["price_per_kwh"]), 2),
            "pue_baseline": round(float(((R["it_kw"] + R["cooling_kw_baseline"]) / R["it_kw"]).mean()), 3),
            "pue_ai": round(float(((R["it_kw"] + R["cooling_kw_ai"]) / R["it_kw"]).mean()), 3),
            "time_above_threshold_baseline_pct": round(float(100 * (R["temp_max"] > thr).mean()), 2),
            "time_above_threshold_ai_pct": round(float(100 * (R["temp_max_ai"] > thr).mean()), 2),
            "hot_threshold_c": round(float(thr), 2),
        }
        return summary, R
