"""
Frontend – Efficacité énergétique des centres de données
Lit les fichiers produits par le notebook (dossier outputs/) :
  - metrics.json
  - test_control_log.csv
  - rf_hvac_controller.joblib   (optionnel : simulateur de décision en direct)

Lancer :  streamlit run app.py
"""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Refroidissement IA – Datacenter", page_icon="🌡️", layout="wide")

# ---- Paramètres identiques à ceux du notebook (section CONFIG) ----
RF_FEATS = ["cpu_now", "cpu_pred", "hot_prob", "temp_mean", "temp_max", "hum", "hour_sin", "hour_cos"]
SP_MIN, SP_MAX = 16.0, 26.0
TEAL, GREY, RED = "#0E6B6B", "#8A9A9E", "#C0392B"


def cop(t):
    """Coefficient de performance du CRAC en fonction de la température de soufflage."""
    return 0.0068 * t ** 2 + 0.0008 * t + 0.458


# ---------------------------------------------------------------- chargement
def find_dir():
    for d in ["./outputs", "../outputs", "/kaggle/working"]:
        if os.path.exists(os.path.join(d, "metrics.json")):
            return d
    return None


@st.cache_data(show_spinner=False)
def load_files(folder):
    with open(os.path.join(folder, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    log = pd.read_csv(os.path.join(folder, "test_control_log.csv"), index_col=0, parse_dates=True)
    return metrics, log


@st.cache_resource(show_spinner=False)
def load_rf(folder):
    path = os.path.join(folder, "rf_hvac_controller.joblib")
    if not os.path.exists(path):
        return None
    import joblib
    return joblib.load(path)


with st.sidebar:
    st.header("Données")
    folder = st.text_input("Dossier des résultats", value=find_dir() or "./outputs")
    st.caption("Dossier généré par le notebook (metrics.json, test_control_log.csv).")

if not os.path.exists(os.path.join(folder, "metrics.json")) or \
        not os.path.exists(os.path.join(folder, "test_control_log.csv")):
    st.title("Refroidissement piloté par IA")
    st.error(f"Fichiers introuvables dans « {folder} ».")
    st.markdown(
        "Exécutez le notebook `datacenter_cooling_ai.ipynb`, puis copiez le dossier `outputs/` "
        "à côté de `app.py` (ou indiquez son chemin dans la barre latérale)."
    )
    st.stop()

metrics, log = load_files(folder)
rf = load_rf(folder)

# ---------------------------------------------------------------- en-tête
st.title("Refroidissement d'un datacenter piloté par IA")
st.write("Prévision de charge (LSTM), détection de points chauds (CNN) et consigne de climatisation "
         "(Random Forest), comparées à une consigne fixe.")

if metrics.get("synthetic_data", False):
    st.warning("Ces résultats viennent du **jeu de données synthétique** du notebook, "
               "pas des données Kaggle. Ils illustrent le fonctionnement, pas un site réel.")

tab_over, tab_ctrl, tab_perf, tab_whatif, tab_live = st.tabs(
    ["Résultats", "Pilotage", "Performance des modèles", "Et si…", "Décision en direct"]
)

# ---------------------------------------------------------------- 1. résultats
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Énergie de refroidissement économisée", f"{metrics['energy_saved_kWh']:,.1f} kWh",
              f"{metrics['energy_saved_pct']:.2f} %")
    c2.metric("CO₂ évité", f"{metrics['CO2_avoided_kg']:,.1f} kg")
    c3.metric("Coût économisé", f"{metrics['cost_saved']:,.2f}")
    c4.metric("PUE", f"{metrics['PUE_AI']:.3f}", f"{metrics['PUE_AI'] - metrics['PUE_baseline']:+.3f} vs fixe",
              delta_color="inverse")
    st.caption(f"Période de test : {metrics['test_days']} jours. Estimation basée sur un modèle énergétique simplifié.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(
            x=["Consigne fixe", "Pilotage IA"],
            y=[metrics["cooling_kWh_baseline"], metrics["cooling_kWh_AI"]],
            marker_color=[GREY, TEAL],
            text=[f"{metrics['cooling_kWh_baseline']:,.0f}", f"{metrics['cooling_kWh_AI']:,.0f}"],
            textposition="outside"))
        fig.update_layout(title="Énergie de refroidissement (kWh)", height=340,
                          margin=dict(t=50, b=10), yaxis_rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = go.Figure(go.Bar(
            x=["Consigne fixe", "Pilotage IA"],
            y=[metrics["time_above_hot_threshold_baseline_pct"], metrics["time_above_hot_threshold_AI_pct"]],
            marker_color=[GREY, TEAL],
            text=[f"{metrics['time_above_hot_threshold_baseline_pct']:.2f} %",
                  f"{metrics['time_above_hot_threshold_AI_pct']:.2f} %"],
            textposition="outside"))
        fig.update_layout(title="Temps au-dessus du seuil de point chaud (%)", height=340,
                          margin=dict(t=50, b=10), yaxis_rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
    st.caption("Le taux de dépassement est calculé avec le même modèle thermique simplifié que celui "
               "qui définit la consigne idéale : il ne prouve pas la sécurité sur un site réel.")

# ---------------------------------------------------------------- 2. pilotage
with tab_ctrl:
    days = max(1, int(np.ceil((log.index[-1] - log.index[0]).total_seconds() / 86400)))
    n_days = st.slider("Jours affichés", 1, days, min(3, days))
    sub = log[log.index < log.index[0] + pd.Timedelta(days=n_days)]
    base_sp = metrics.get("BASE_SETPOINT", 20.0)

    fig = go.Figure(go.Scatter(x=sub.index, y=sub["cpu_load"], line=dict(color=TEAL), name="Charge CPU"))
    fig.update_layout(title="Charge CPU des serveurs (0–1)", height=250, margin=dict(t=45, b=10))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=sub.index, y=sub["setpoint_AI"], name="Consigne IA", line=dict(color=TEAL))
    fig.add_hline(y=base_sp, line_dash="dash", line_color=RED, annotation_text="Consigne fixe")
    fig.update_layout(title="Consigne de soufflage (°C)", height=270, margin=dict(t=45, b=10))
    st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_scatter(x=sub.index, y=sub["cooling_kW_baseline"], name="Consigne fixe", line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=sub.index, y=sub["cooling_kW_AI"], name="Pilotage IA", line=dict(color=TEAL))
    fig.update_layout(title="Puissance de refroidissement (kW)", height=270, margin=dict(t=45, b=10))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Voir le journal complet"):
        st.dataframe(log, width="stretch")
        st.download_button("Télécharger en CSV", log.to_csv().encode("utf-8"), "test_control_log.csv", "text/csv")

# ---------------------------------------------------------------- 3. performance
with tab_perf:
    a, b, c = st.columns(3)
    with a:
        st.subheader("LSTM – charge CPU")
        st.dataframe(pd.DataFrame({
            "LSTM": [metrics.get("LSTM_MAE"), metrics.get("LSTM_RMSE")],
            "Persistance": [metrics.get("Naive_MAE"), metrics.get("Naive_RMSE")]},
            index=["MAE", "RMSE"]).round(4), width="stretch")
        if metrics.get("Naive_RMSE"):
            gain = 100 * (1 - metrics["LSTM_RMSE"] / metrics["Naive_RMSE"])
            st.caption(f"RMSE {gain:.1f} % plus bas que la persistance.")
    with b:
        st.subheader("CNN – point chaud")
        st.dataframe(pd.DataFrame({
            "CNN": [metrics.get("CNN_AUC"), metrics.get("CNN_F1")],
            "Persistance": [None, metrics.get("Persistence_F1")]},
            index=["AUC", "F1"]).round(3), width="stretch")
    with c:
        st.subheader("Random Forest – consigne")
        st.dataframe(pd.DataFrame({"Valeur": [metrics.get("RF_MAE_degC"), metrics.get("RF_R2")]},
                                  index=["MAE (°C)", "R²"]).round(3), width="stretch")
        st.caption("Le RF imite une politique idéale calculée avec les températures futures.")

# ---------------------------------------------------------------- 4. et si
with tab_whatif:
    st.write("Recalcule les économies à partir du journal de test, avec d'autres hypothèses.")
    w1, w2 = st.columns(2)
    n_serv = w1.slider("Nombre de serveurs", 50, 2000, 200, 50)
    p_idle = w1.slider("Puissance d'un serveur au repos (kW)", 0.05, 0.30, 0.10, 0.01)
    p_max = w1.slider("Puissance d'un serveur à pleine charge (kW)", 0.15, 0.80, 0.30, 0.01)
    base = w2.slider("Consigne fixe de référence (°C)", 16.0, 26.0, 20.0, 0.5)
    co2 = w2.slider("Facteur d'émission (kg CO₂/kWh)", 0.0, 1.0, 0.45, 0.01)
    price = w2.slider("Prix du kWh", 0.01, 0.50, 0.10, 0.01)

    dt_h = (log.index[1] - log.index[0]).total_seconds() / 3600 if len(log) > 1 else 5 / 60
    p_it = n_serv * (p_idle + (p_max - p_idle) * log["cpu_load"].values)
    k_base = (p_it / cop(base)).sum() * dt_h
    k_ai = (p_it / cop(log["setpoint_AI"].values)).sum() * dt_h
    saved = k_base - k_ai

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Consigne fixe", f"{k_base:,.0f} kWh")
    m2.metric("Pilotage IA", f"{k_ai:,.0f} kWh")
    m3.metric("Économie", f"{saved:,.0f} kWh", f"{100 * saved / k_base:.2f} %")
    m4.metric("CO₂ évité / coût", f"{saved * co2:,.0f} kg", f"{saved * price:,.1f}", delta_color="off")
    if p_max <= p_idle:
        st.warning("La puissance à pleine charge doit dépasser celle au repos.")
    st.caption("Le journal garde les consignes IA du notebook : seule l'estimation d'énergie change.")

# ---------------------------------------------------------------- 5. décision en direct
with tab_live:
    if rf is None:
        st.info("Ajoutez `rf_hvac_controller.joblib` dans le dossier des résultats pour activer ce simulateur.")
    else:
        st.write("Réglez les entrées du contrôleur et lisez la consigne choisie par le Random Forest.")
        l1, l2 = st.columns(2)
        cpu_now = l1.slider("Charge CPU actuelle", 0.0, 1.0, 0.5, 0.01)
        cpu_pred = l1.slider("Charge CPU prévue (LSTM, 30 min)", 0.0, 1.0, 0.55, 0.01)
        hot_prob = l1.slider("Probabilité de point chaud (CNN)", 0.0, 1.0, 0.10, 0.01)
        hour = l1.slider("Heure de la journée", 0.0, 23.75, 14.0, 0.25)
        t_mean = l2.slider("Température moyenne des racks (°C)", 15.0, 35.0, 23.0, 0.1)
        t_max = l2.slider("Température maximale des racks (°C)", 15.0, 40.0, 26.0, 0.1)
        hum = l2.slider("Humidité (%)", 15.0, 80.0, 45.0, 1.0)
        safety = l2.checkbox("Couche de sécurité (risque > 0,8 → refroidir plus fort)", value=True)

        row = {"cpu_now": cpu_now, "cpu_pred": cpu_pred, "hot_prob": hot_prob, "temp_mean": t_mean,
               "temp_max": max(t_max, t_mean), "hum": hum,
               "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24)}
        sp = float(rf.predict(pd.DataFrame([row])[RF_FEATS])[0])
        raw = sp
        if safety and hot_prob > 0.8:
            sp = min(sp, 20.0 - 2.0)
        sp = float(np.clip(sp, SP_MIN, SP_MAX))

        r1, r2, r3 = st.columns(3)
        r1.metric("Consigne choisie", f"{sp:.1f} °C", f"{sp - 20.0:+.1f} vs fixe (20 °C)", delta_color="off")
        r2.metric("Prédiction brute du RF", f"{raw:.1f} °C")
        p_it1 = 200 * (0.10 + 0.20 * cpu_now)
        r3.metric("Refroidissement estimé", f"{p_it1 / cop(sp):.1f} kW", f"{p_it1 / cop(sp) - p_it1 / cop(20.0):+.1f} kW vs fixe",
                  delta_color="inverse")
        st.caption("Estimation pour 200 serveurs (0,10 à 0,30 kW chacun), comme dans le notebook.")
