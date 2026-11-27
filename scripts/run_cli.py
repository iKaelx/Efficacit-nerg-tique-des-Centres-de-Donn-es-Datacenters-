"""Command-line use of the models (no Streamlit): load a .mpk, control a series, print the decision and savings.

Examples (from the project root):
    python scripts/run_cli.py                                   # newest package in models/ + its bundled replay
    python scripts/run_cli.py --csv data/my_room.csv            # your own series
    python scripts/run_cli.py --model models/datacenter_cooling_models.mpk --out outputs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cooling_ai import CoolingEngine, load_csv   # noqa: E402


def newest_package() -> Path:
    pkgs = sorted((ROOT / "models").glob("*.mpk"), key=lambda p: p.stat().st_mtime)
    if not pkgs:
        sys.exit("No .mpk package in models/. Train the notebook first (section 7b) and copy the file there.")
    return pkgs[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="path to the .mpk package (default: newest in models/)")
    ap.add_argument("--csv", type=Path, help="series to control (default: the replay bundled in the package)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs", help="folder for control_log.csv and report.json")
    args = ap.parse_args()

    model_path = args.model or newest_package()
    print(f"Installing {model_path.name} ...")
    engine = CoolingEngine.from_mpk(model_path)
    if engine.synthetic:
        print("WARNING: this package was trained on SYNTHETIC data; results are illustrative only.")

    df = load_csv(args.csv, engine.meta) if args.csv else engine.demo_data()
    if df is None:
        sys.exit("The package has no bundled replay; pass --csv.")

    res = engine.run(df)
    summary, log = engine.energy_report(res)

    last = engine.decide(df)                                   # what to do RIGHT NOW (last time step of the series)
    print("\nLatest decision:")
    print(json.dumps(last.as_dict(), indent=2))
    print("\nEnergy report vs fixed setpoint:")
    print(json.dumps(summary, indent=2))

    args.out.mkdir(parents=True, exist_ok=True)
    log.to_csv(args.out / "control_log.csv")
    (args.out / "report.json").write_text(json.dumps({"latest_decision": last.as_dict(), "energy": summary}, indent=2))
    print(f"\nSaved {args.out / 'control_log.csv'} and {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
