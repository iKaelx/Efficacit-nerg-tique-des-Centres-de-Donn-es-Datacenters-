# AI-driven data-center cooling

Reduce the electricity bill and carbon footprint of a server room by controlling the HVAC supply-air setpoint.

| Module | Technique | Role |
|---|---|---|
| Forecast | LSTM | Predicts the servers' CPU load for the next 30 min |
| Thermal analysis | CNN | Reads the 2D rack heat map, outputs a hotspot probability |
| Control | Random Forest + safety layer | Chooses the supply-air setpoint (cools harder if hotspot risk > 80 %) |
| Evaluation | Energy model | kWh, cost, CO₂ and PUE vs a fixed setpoint |

## Project layout

```
notebooks/datacenter_cooling_ai.ipynb   train the 3 models, export ONE .mpk package
models/                                 put the .mpk here (demo_synthetic.mpk = fake-data demo, replace it)
src/cooling_ai/engine.py                loads the .mpk and runs LSTM + CNN + RF (the reusable code)
app/streamlit_app.py                    Streamlit front-end
scripts/run_cli.py                      same engine from the command line
tests/test_engine.py                    pytest smoke tests
data/                                   put the two Kaggle CSV files here when training locally
.vscode/                                launch (F5), tasks and settings
```

## Workflow

1. **Setup (VS Code)**: open this folder, create a virtual environment, then
   `python -m pip install -r requirements.txt`. Select it as the Python interpreter and as the notebook kernel.
2. **Train**: open `notebooks/datacenter_cooling_ai.ipynb` and *Run All*.
   - *Kaggle (GPU)*: Add Input → **Server Workload Traces** + **Data Center Temperature & Humidity Sensors**.
     Download `datacenter_cooling_models.mpk` from `/kaggle/working`.
   - *Local*: put both CSVs in `data/`; the `.mpk` is written straight to `models/`.
   - If a dataset is missing the notebook stops with an explicit error (`REQUIRE_REAL_DATA = True`),
     so you never train on synthetic data by accident.
3. **Install the models**: copy the `.mpk` into `models/` (or upload it in the app sidebar). That is the whole "installation".
4. **Run the front-end**: `streamlit run app/streamlit_app.py` (or F5 → *Streamlit app*).
5. **Or use the code directly**:

```python
from cooling_ai import CoolingEngine, load_csv

engine = CoolingEngine.from_mpk("models/datacenter_cooling_models.mpk")
df = load_csv("data/my_room.csv", engine.meta)     # time, cpu, T_0..T_n, optional hum
print(engine.decide(df).as_dict())                 # setpoint to apply NOW
summary, log = engine.energy_report(engine.run(df))  # savings over the whole series
```

CLI: `python scripts/run_cli.py --csv data/my_room.csv` writes `outputs/control_log.csv` and `outputs/report.json`.

## CSV format expected by the app / CLI

A time column (`time`, `timestamp`, `date`...), `cpu` (0-1 or 0-100 %), the sensor columns
`T_0 ... T_{n-1}` (same count as the trained package, shown in the app's *Models and data* tab) and optionally `hum`.
Data is resampled to the model's time step (5 min) and needs at least 37 steps.

## Honest limits (state them in the report)

- The public datasets contain no record of real HVAC actions. The Random Forest imitates an **oracle policy**
  computed with future temperatures; at run time it only sees information available now.
- The two datasets come from different sources: their time axes are aligned artificially and the sensors are laid out
  on the heat-map grid row by row.
- Energy, cost and CO₂ figures come from a **simplified model** (server power, CRAC efficiency curve, thermal coupling `K_PASS`),
  not from measurements. Validate with real metering before quoting them.
- `models/demo_synthetic.mpk` was trained on **synthetic** data to demonstrate the workflow; the app shows a warning while it is selected.
- `.mpk` files contain a pickled scikit-learn model: only install packages you trust.
