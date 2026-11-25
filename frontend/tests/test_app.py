import csv
import json
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.output_dir = Path(cls.temp_dir.name)
        cls.invalid_dir = cls.output_dir / "invalid"
        cls.invalid_dir.mkdir()
        metrics = {
            "energy_saved_kWh": 1.0,
            "energy_saved_pct": 1.0,
            "CO2_avoided_kg": 0.45,
            "cost_saved": 0.1,
            "PUE_AI": 1.2,
            "PUE_baseline": 1.21,
            "test_days": 1.0,
            "cooling_kWh_baseline": 100.0,
            "cooling_kWh_AI": 99.0,
            "time_above_hot_threshold_baseline_pct": 1.0,
            "time_above_hot_threshold_AI_pct": 0.0,
        }
        (cls.output_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (cls.invalid_dir / "metrics.json").write_text("{}", encoding="utf-8")
        with (cls.output_dir / "test_control_log.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["timestamp", "cpu_load", "setpoint_AI", "cooling_kW_baseline", "cooling_kW_AI"]
            )
            writer.writerow(["2024-01-01 00:00:00", 0.5, 21.0, 10.0, 9.0])
            writer.writerow(["2024-01-01 00:05:00", 0.6, 21.0, 11.0, 10.0])
        (cls.invalid_dir / "test_control_log.csv").write_bytes(
            (cls.output_dir / "test_control_log.csv").read_bytes()
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def load_app(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        app.text_input[0].set_value(str(self.output_dir))
        return app.run(timeout=60)

    def test_dashboard_loads_valid_results(self):
        app = self.load_app()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("PUE", [metric.label for metric in app.get("metric")])
        self.assertTrue(any(tab.label == "Résultats" for tab in app.get("tab")))

    def test_invalid_server_power_does_not_show_energy_totals(self):
        app = self.load_app()
        sliders = app.get("slider")
        idle = next(slider for slider in sliders if "repos" in slider.label)
        full = next(slider for slider in sliders if "pleine charge" in slider.label)
        idle.set_value(0.25)
        full.set_value(0.15)
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("doit dépasser celle au repos" in warning.value for warning in app.warning)
        )
        what_if_metrics = {"Consigne fixe", "Pilotage IA", "Économie"}
        self.assertTrue(what_if_metrics.isdisjoint(metric.label for metric in app.get("metric")))

    def test_incomplete_metrics_are_reported_without_crashing(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=60)
        app.text_input[0].set_value(str(self.invalid_dir))
        app.run(timeout=60)

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("metrics.json is missing" in error.value for error in app.error))


if __name__ == "__main__":
    unittest.main()
