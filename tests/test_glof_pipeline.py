"""
Unit & Integration Test Suite for Cascading GLOF Hydro-Deep Learning System.
"""

import os
import sys
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_builder import BasinGraphBuilder
from src.physics.synthetic_generator import HydrodynamicGLOFSimulator
from src.models.breach_predictor import BreachPredictor
from src.models.flood_gnn import RiverGNN
from src.pipeline.early_warning import GLOFEarlyWarningSystem

class TestGLOFPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = "data/processed"
        cls.model_dir = "models"
        os.makedirs(cls.data_dir, exist_ok=True)
        os.makedirs(cls.model_dir, exist_ok=True)

    def test_01_graph_builder(self):
        builder = BasinGraphBuilder(basin_name="Test_Basin")
        node_feats, edge_idx, edge_attr = builder.build_synthetic_himalayan_reach(num_nodes=50)

        self.assertEqual(node_feats.shape, (50, 8))
        self.assertEqual(edge_idx.shape[0], 2)
        self.assertFalse(np.isnan(node_feats).any())
        self.assertTrue(np.all(node_feats[:, 0] >= 0.0)) # Positive elevation
        self.assertTrue(np.all(node_feats[:, 7] >= 1.0)) # Bulking factor >= 1.0x

    def test_02_hydrodynamic_simulation(self):
        sim = HydrodynamicGLOFSimulator(data_dir=self.data_dir, timesteps=36, dt_minutes=5.0)
        h_mat, arr_t, h_max, v_pk = sim.simulate_single_breach(
            Q_peak=3000.0,
            breach_width=80.0,
            base_flow=40.0
        )

        self.assertEqual(h_mat.shape[0], sim.num_nodes)
        self.assertEqual(h_mat.shape[1], 36)
        self.assertTrue(np.all(h_max > 0.0))
        # Wave arrival time at outlet should be greater than at inlet
        self.assertGreater(arr_t[-1], arr_t[0])

    def test_03_stage1_breach_predictor(self):
        predictor = BreachPredictor(model_dir=self.model_dir)
        sample_input = {
            "cumulative_melt_14d": 120.0,
            "rainfall_48h": 85.0,
            "lake_area_change_pct": 18.0,
            "slope_angle_deg": 45.0,
            "moraine_creep_velocity_mmyr": 50.0,
            "cascading_proximity_index": 0.85
        }
        res = predictor.predict(sample_input)
        self.assertIn("breach_probability", res)
        self.assertIn("predicted_q_peak_m3s", res)
        self.assertIn("alert_level", res)
        self.assertGreaterEqual(res["breach_probability"], 0.0)
        self.assertLessEqual(res["breach_probability"], 1.0)
        self.assertGreater(res["predicted_q_peak_m3s"], 0.0)

    def test_04_stage2_river_gnn_forward(self):
        model = RiverGNN(in_channels=8, hidden_dim=32, out_timesteps=36)
        x = torch.randn(40, 8)
        edge_index = torch.tensor([[i, i+1] for i in range(39)], dtype=torch.long).t()

        depth_pred, arrival_pred = model(x, edge_index, breach_pulse=1500.0)

        self.assertEqual(depth_pred.shape, (40, 36))
        self.assertEqual(arrival_pred.shape, (40,))
        self.assertTrue(torch.all(depth_pred >= 0.0)) # Softplus guarantee
        self.assertTrue(torch.all(arrival_pred >= 0.0))

    def test_05_early_warning_system(self):
        ews = GLOFEarlyWarningSystem(data_dir=self.data_dir, model_dir=self.model_dir)
        test_obs = {
            "cumulative_melt_14d": 140.0,
            "rainfall_48h": 90.0,
            "lake_area_change_pct": 20.0,
            "slope_angle_deg": 50.0,
            "moraine_creep_velocity_mmyr": 60.0,
            "cascading_proximity_index": 0.90
        }
        hazard = ews.assess_realtime_hazard(test_obs)
        self.assertIn("upstream_lake_status", hazard)
        self.assertIn("community_bulletin", hazard)
        self.assertGreater(len(hazard["community_bulletin"]), 0)

    def test_06_emergency_alert_dispatch(self):
        ews = GLOFEarlyWarningSystem(data_dir=self.data_dir, model_dir=self.model_dir)
        test_obs = {
            "cumulative_melt_14d": 145.0,
            "rainfall_48h": 95.0,
            "lake_area_change_pct": 25.0,
            "slope_angle_deg": 52.0,
            "moraine_creep_velocity_mmyr": 65.0,
            "cascading_proximity_index": 0.95
        }
        hazard = ews.assess_realtime_hazard(test_obs)
        alert_payload = ews.dispatch_emergency_alert(hazard)
        self.assertIn("protocol", alert_payload)
        self.assertIn("alert_status", alert_payload)
        self.assertIn("evacuation_targets", alert_payload)
        self.assertEqual(alert_payload["dispatch_status"], "SENT_CONFIRMED")

if __name__ == "__main__":
    unittest.main()
