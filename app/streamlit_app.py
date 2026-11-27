"""Streamlit front-end for the AI data-center cooling controller.

Run from the project root:   streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))                      # lets the app import cooling_ai without installing it

import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402
import plotly.graph_objects as go                           # noqa: E402
import streamlit as st                                      # noqa: E402

from cooling_ai import CoolingEngine, load_csv              # noqa: E402

MODELS_DIR = ROOT / "models"
AI, BASE, ALERT = "#14919B", "#E07A3F", "#D64550"          # AI controller / fixed baseline / hotspot

st.set_page_config(page_title="AI cooling control", page_icon="❄️", layout="wide")


# --------------------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Installing the models (LSTM, CNN, Random Forest)...")
def engine_from_path(path: str, mtime: float) -> CoolingEngine:
    return CoolingEngine.from_mpk(path)


@st.cache_resource(show_spinner="Installing the uploaded .mpk package...")
def engine_from_upload(digest: str, _blob: bytes) -> CoolingEngine:
    return CoolingEngine.from_mpk(_blob)


@st.cache_data(show_spinner="Running LSTM + CNN + Random Forest over the series...")
def run_pipeline(_engine: CoolingEngine, engine_key: str, _df: pd.DataFrame, data_key: str) -> pd.DataFrame:
    return _engine.run(_df)


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.14, x=0), hovermode="x unified")
    return fig


# --------------------------------------------------------------------------- sidebar: model + data
with st.sidebar:
    st.header("Model package")
    uploaded_pkg = st.file_uploader("Install a .mpk package", type=["mpk"],
                                    help="Exported by the training notebook (section 7b).")
    local_pkgs = sorted(MODELS_DIR.glob("*.mpk"))
    try:
        if uploaded_pkg is not None:
            blob = uploaded_pkg.getvalue()
            digest = hashlib.md5(blob).hexdigest()
            engine, engine_key = engine_from_upload(digest, blob), f"upload-{digest}"
            st.caption(f"Installed from upload: {uploaded_pkg.name}")
        elif local_pkgs:
            chosen = st.selectbox("Packages in models/", local_pkgs, format_func=lambda p: p.name)
            engine = engine_from_path(str(chosen), chosen.stat().st_mtime)
            engine_key = f"{chosen.name}-{chosen.stat().st_mtime}"
        else:
            st.error("No .mpk package found in `models/`. Train the notebook and drop the file there, "
                     "or upload one above.")
            st.stop()
    except Exception as exc:                                 # corrupted file, wrong format, TF error...
        st.error(f"Could not install the package: {exc}")
        st.stop()
    meta = engine.meta

    st.header("Data")
    demo_df = engine.demo_data()
    options = (["Replay bundled in the package"] if demo_df is not None else []) + ["Upload a CSV"]
    source = st.radio("Series to control", options, label_visibility="collapsed")
    if source == "Upload a CSV":
        csv_file = st.file_uploader("CSV with time, cpu, " + f"{meta['sensor_cols'][0]}..{meta['sensor_cols'][-1]}",
                                    type=["csv"])
        if csv_file is None:
            st.info("Waiting for a CSV. Columns: a time column, `cpu` (0-1 or %), "
                    f"{meta['n_sensors']} sensor columns `T_0`..`T_{meta['n_sensors'] - 1}`, optional `hum`.")
            st.stop()
        csv_bytes = csv_file.getvalue()
        try:
            df = load_csv(pd.io.common.BytesIO(csv_bytes), meta)
            engine.prepare(df)                               # validates the columns early
        except Exception as exc:
            st.error(f"Invalid CSV: {exc}")
            st.stop()
        data_key = hashlib.md5(csv_bytes).hexdigest()
    else:
        df, data_key = demo_df, "replay"

    with st.expander("Energy assumptions", expanded=False):
        e = meta["energy"]
        params = {
            "n_servers": st.number_input("Servers in the room", 1, 100_000, int(e["n_servers"])),
            "p_idle_kw": st.number_input("Idle power per server (kW)", 0.0, 5.0, float(e["p_idle_kw"]), 0.01),
            "p_max_kw": st.number_input("Full-load power per server (kW)", 0.0, 5.0, float(e["p_max_kw"]), 0.01),
            "base_setpoint_C": st.number_input("Fixed baseline setpoint (°C)", 10.0, 30.0,
                                               float(meta["base_setpoint_C"]), 0.5),
            "co2_kg_per_kwh": st.number_input("Grid emissions (kg CO₂/kWh)", 0.0, 2.0, float(e["co2_kg_per_kwh"]), 0.01),
            "price_per_kwh": st.number_input("Electricity price (per kWh)", 0.0, 5.0, float(e["price_per_kwh"]), 0.01),
        }

# --------------------------------------------------------------------------- run the pipeline once
res = run_pipeline(engine, engine_key, df, data_key)
valid = res[res["setpoint_ai"].notna()]
if valid.empty:
    st.error(f"Not enough data: at least {meta['lookback'] + 1} time steps of {meta['step']} are needed.")
    st.stop()
summary, R = engine.energy_report(res, params)
base_sp, thr = params["base_setpoint_C"], meta["hot_threshold_C"]
step = pd.Timedelta(meta["step"])
sensors = meta["sensor_cols"]

st.title("AI-driven cooling control")
st.write("The LSTM forecasts the servers' CPU load, the CNN reads the rack heat map for hotspots, "
         "and the Random Forest turns both into an HVAC supply-air setpoint.")
if engine.synthetic:
    st.warning("This package was trained on **synthetic demo data**. The figures below illustrate the workflow "
               "only; retrain the notebook on the real datasets for meaningful results.")

tab_live, tab_save, tab_what, tab_info = st.tabs(["Live decision", "Savings", "What-if", "Models and data"])

# --------------------------------------------------------------------------- tab 1: live decision
with tab_live:
    n_valid = len(valid)
    if st.session_state.get("_data_key") != data_key:        # new data -> reset the time cursor
        st.session_state["_data_key"] = data_key
        st.session_state["step_idx"] = min(n_valid - 1, n_valid // 3)
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Calmest moment", width="stretch"):
        st.session_state["step_idx"] = int(np.argmin(valid["hot_prob"].to_numpy()))
    if b2.button("Highest hotspot risk", width="stretch"):
        st.session_state["step_idx"] = int(np.argmax(valid["hot_prob"].to_numpy()))
    idx = st.select_slider("Moment to control", options=list(range(n_valid)), key="step_idx",
                           format_func=lambda k: valid.index[k].strftime("%a %d %b, %H:%M"))
    ts = valid.index[idx]
    dec = engine._decision_from_row(valid.iloc[idx], ts)
    pw = R.loc[ts]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("HVAC setpoint (AI)", f"{dec.setpoint_c:.1f} °C", f"{dec.setpoint_c - base_sp:+.1f} °C vs fixed {base_sp:.0f} °C")
    m2.metric("Hotspot risk (next 30 min)", f"{dec.hot_prob:.0%}", help="CNN probability that the hottest rack crosses the threshold.")
    m3.metric("CPU load now", f"{dec.cpu_now:.0%}", f"{dec.cpu_forecast.mean() - dec.cpu_now:+.0%} forecast (30 min avg)")
    m4.metric("Cooling power (AI)", f"{pw['cooling_kw_ai']:.1f} kW",
              f"{100 * (pw['cooling_kw_ai'] / pw['cooling_kw_baseline'] - 1):+.1f}% vs baseline", delta_color="inverse")

    if dec.safety_override:
        st.error(f"Safety layer active: hotspot risk is above {meta['hot_prob_safety_limit']:.0%}, so the supply air "
                 f"is lowered to {dec.setpoint_c:.1f} °C (the Random Forest alone proposed {dec.rf_setpoint_c:.1f} °C).")
    elif dec.hot_prob > 0.5:
        st.warning("Elevated hotspot risk: the controller keeps the supply air cooler than usual.")
    elif dec.setpoint_c > base_sp:
        st.success("Thermal margin available: the controller runs warmer supply air than the baseline to save energy.")
    else:
        st.info("The controller stays at or below the baseline setpoint to protect the racks.")

    left, right = st.columns(2)
    pos = res.index.get_loc(ts)
    hist = res["cpu_now"].iloc[max(0, pos - meta["lookback"] + 1): pos + 1]
    fx = [ts] + [ts + step * (k + 1) for k in range(meta["horizon"])]
    fy = [dec.cpu_now] + list(dec.cpu_forecast)
    fig = go.Figure()
    fig.add_scatter(x=hist.index, y=hist.values, name="CPU load (history)", line=dict(color=BASE, width=2))
    fig.add_scatter(x=fx, y=fy, name="LSTM forecast", line=dict(color=AI, width=3, dash="dot"), mode="lines+markers")
    fig.update_yaxes(title="CPU load", tickformat=".0%")
    left.plotly_chart(style(fig.update_layout(title="Load forecast")), key="fig_cpu")

    grid = engine.to_grid(df.loc[ts, sensors].to_numpy(dtype="float32"))[0]
    hot_y, hot_x = np.where(grid > thr)
    fig = go.Figure(go.Heatmap(z=grid, colorscale="YlOrRd", zmin=float(df[sensors].min().min()),
                               zmax=float(df[sensors].max().max()), colorbar=dict(title="°C")))
    if len(hot_x):
        fig.add_scatter(x=hot_x, y=hot_y, mode="markers", name=f"above {thr:.1f} °C",
                        marker=dict(symbol="x", size=12, color=ALERT, line=dict(width=1)))
    fig.update_yaxes(autorange="reversed", showticklabels=False, scaleanchor="x")
    fig.update_xaxes(showticklabels=False)
    right.plotly_chart(style(fig.update_layout(title=f"Rack heat map (threshold {thr:.1f} °C)", hovermode="closest")),
                       key="fig_heat")

    lo, hi = max(0, idx - 144), min(n_valid, idx + 144)       # +/- 12 h around the chosen moment
    win = valid.iloc[lo:hi]
    fig = go.Figure()
    fig.add_scatter(x=win.index, y=win["setpoint_ai"], name="AI setpoint", line=dict(color=AI, width=2))
    fig.add_hline(y=base_sp, line=dict(color=BASE, dash="dash"), annotation_text="fixed baseline")
    fig.add_shape(type="line", x0=ts.to_pydatetime(), x1=ts.to_pydatetime(), y0=0, y1=1, yref="paper",
                  line=dict(color="gray", width=1, dash="dot"))
    fig.update_yaxes(title="Supply air (°C)")
    st.plotly_chart(style(fig.update_layout(title="Setpoint around this moment (±12 h)"), 280), key="fig_sp")

    with st.expander("How this decision was made"):
        st.markdown(
            f"1. **LSTM**: the average CPU load over the next {meta['horizon'] * pd.Timedelta(meta['step']).seconds // 60} "
            f"minutes is forecast at **{dec.cpu_forecast.mean():.0%}** (now {dec.cpu_now:.0%}).\n"
            f"2. **CNN**: the rack heat map gives a **{dec.hot_prob:.0%}** hotspot probability.\n"
            f"3. **Random Forest**: from forecast, risk, temperatures, humidity and hour of day it proposes "
            f"**{dec.rf_setpoint_c:.1f} °C**.\n"
            f"4. **Safety layer**: " + ("risk above the limit, so the setpoint is capped at "
                                         f"{base_sp - 2:.0f} °C." if dec.safety_override else "risk below the limit, no override.") +
            f"\n\nFinal setpoint: **{dec.setpoint_c:.1f} °C**.")

# --------------------------------------------------------------------------- tab 2: savings
with tab_save:
    st.subheader(f"AI controller vs fixed {base_sp:.0f} °C setpoint, over {summary['days']} days")
    a, b, c, d = st.columns(4)
    a.metric("Cooling energy saved", f"{summary['energy_saved_kwh']:,.0f} kWh", f"{summary['energy_saved_pct']:.1f}%")
    b.metric("Cost saved", f"{summary['cost_saved']:,.2f}")
    c.metric("CO₂ avoided", f"{summary['co2_avoided_kg']:,.0f} kg")
    d.metric("PUE", f"{summary['pue_ai']:.3f}", f"{summary['pue_ai'] - summary['pue_baseline']:+.3f} vs {summary['pue_baseline']:.3f}",
             delta_color="inverse")
    e1, e2 = st.columns(2)
    e1.metric("Time above hot threshold, baseline", f"{summary['time_above_threshold_baseline_pct']:.2f}%")
    e2.metric("Time above hot threshold, AI", f"{summary['time_above_threshold_ai_pct']:.2f}%",
              f"{summary['time_above_threshold_ai_pct'] - summary['time_above_threshold_baseline_pct']:+.2f} pts",
              delta_color="inverse")

    fig = go.Figure()
    fig.add_scatter(x=R.index, y=R["setpoint_ai"], name="AI setpoint", line=dict(color=AI, width=1.5))
    fig.add_hline(y=base_sp, line=dict(color=BASE, dash="dash"), annotation_text="fixed baseline")
    fig.update_yaxes(title="Supply air (°C)")
    st.plotly_chart(style(fig.update_layout(title="HVAC setpoint")), key="fig_s1")

    fig = go.Figure()
    fig.add_scatter(x=R.index, y=R["cooling_kw_baseline"], name="Baseline", line=dict(color=BASE, width=1.5))
    fig.add_scatter(x=R.index, y=R["cooling_kw_ai"], name="AI", line=dict(color=AI, width=1.5))
    fig.update_yaxes(title="kW")
    st.plotly_chart(style(fig.update_layout(title="Cooling power")), key="fig_s2")

    dt_h = pd.Timedelta(meta["step"]).total_seconds() / 3600
    fig = go.Figure(go.Scatter(x=R.index, y=((R["cooling_kw_baseline"] - R["cooling_kw_ai"]) * dt_h).cumsum(),
                               fill="tozeroy", line=dict(color=AI), name="Saved"))
    fig.update_yaxes(title="kWh")
    st.plotly_chart(style(fig.update_layout(title="Cumulative energy saved"), 280), key="fig_s3")

    st.caption("These figures come from a simplified energy model (server power, CRAC efficiency curve, "
               "thermal coupling between setpoint and rack temperature), not from measurements. "
               "Validate with real metering before quoting them.")
    log = R[["cpu_now", "cpu_pred", "hot_prob", "temp_max", "setpoint_ai", "cooling_kw_baseline", "cooling_kw_ai"]]
    st.download_button("Download control log (CSV)", log.to_csv().encode(), "control_log.csv", "text/csv")

# --------------------------------------------------------------------------- tab 3: what-if
with tab_what:
    st.write("Set the situation by hand and see what the Random Forest and the safety layer would do.")
    med = valid.median(numeric_only=True)
    w1, w2, w3 = st.columns(3)
    cpu_now = w1.slider("CPU load now", 0.0, 1.0, float(np.clip(med["cpu_now"], 0, 1)), 0.01)
    cpu_pred = w1.slider("Forecast CPU load (30 min avg)", 0.0, 1.0, float(np.clip(med["cpu_pred"], 0, 1)), 0.01)
    hot_prob = w2.slider("Hotspot risk (CNN)", 0.0, 1.0, float(np.clip(med["hot_prob"], 0, 1)), 0.01)
    hour = w2.slider("Hour of day", 0.0, 23.75, 12.0, 0.25)
    t_lo, t_hi = float(df[sensors].min().min()) - 2, float(df[sensors].max().max()) + 2
    temp_mean = w3.slider("Mean rack temperature (°C)", t_lo, t_hi, float(med["temp_mean"]), 0.1)
    temp_max = w3.slider("Hottest rack (°C)", t_lo, t_hi, float(max(med["temp_max"], temp_mean)), 0.1)
    hum = w3.slider("Humidity (%)", 5.0, 95.0, float(np.clip(med["hum"], 5, 95)), 0.5)
    sp, rf_sp, override = engine.decide_from_features({
        "cpu_now": cpu_now, "cpu_pred": cpu_pred, "hot_prob": hot_prob, "temp_mean": temp_mean, "temp_max": temp_max,
        "hum": hum, "hour_sin": float(np.sin(2 * np.pi * hour / 24)), "hour_cos": float(np.cos(2 * np.pi * hour / 24))})
    r1, r2 = st.columns(2)
    r1.metric("Recommended setpoint", f"{sp:.1f} °C", f"{sp - base_sp:+.1f} °C vs fixed {base_sp:.0f} °C")
    r2.metric("Random Forest alone", f"{rf_sp:.1f} °C")
    if override:
        st.error("Safety layer active: hotspot risk is above the limit.")

# --------------------------------------------------------------------------- tab 4: info
with tab_info:
    i1, i2 = st.columns(2)
    with i1:
        st.subheader("Package")
        info = {
            "Trained on": "SYNTHETIC demo data" if engine.synthetic else "real datasets",
            "Time step": meta["step"], "LSTM history": f"{meta['lookback']} steps",
            "Forecast horizon": f"{meta['horizon']} steps", "Sensors": meta["n_sensors"],
            "Heat map grid": f"{meta['grid_side']} x {meta['grid_side']}",
            "Hot threshold": f"{meta['hot_threshold_C']:.2f} °C",
            "Setpoint range": f"{meta['setpoint_range_C'][0]:.0f} to {meta['setpoint_range_C'][1]:.0f} °C",
        }
        st.dataframe(pd.DataFrame({"value": info.values()}, index=list(info.keys()), dtype=str))
    with i2:
        st.subheader("Training metrics")
        mets = {k: (f"{v:.4g}" if isinstance(v, float) else str(v)) for k, v in meta.get("metrics", {}).items()}
        st.dataframe(pd.DataFrame({"value": mets.values()}, index=list(mets.keys()), dtype=str))
    st.subheader("Series being controlled")
    st.dataframe(df.head(50))
    st.caption(f"{len(df)} time steps from {df.index[0]} to {df.index[-1]}.")
