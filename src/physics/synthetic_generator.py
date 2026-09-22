"""
Synthetic Ground Truth Generator & Hydrodynamic GLOF Wave Simulator.

Simulates 200-500 breach scenarios using 1D/2D hydraulic wave routing:
- Kinematic & Diffusive flood wave routing with Manning's equation.
- Debris & sediment bulking factor (1.5x - 2.0x volume expansion).
- Variations across:
  * Dam breach widths (30m to 200m)
  * Peak discharge Q_peak (500 m3/s to 10,000 m3/s)
  * Base river flows (monsoon vs. dry season)
- Outputs labels:
  * t_arrival: Wave arrival time (minutes)
  * h_max: Maximum water depth (meters)
  * v_peak: Peak flow velocity (m/s)
  * h(t): Time-varying depth hydrographs [Num_Scenarios, Num_Nodes, Timesteps]
"""

import os
import argparse
import numpy as np
import pandas as pd

class HydrodynamicGLOFSimulator:
    def __init__(self, data_dir="data/processed", timesteps=72, dt_minutes=5.0):
        self.data_dir = data_dir
        self.timesteps = timesteps          # 72 timesteps * 5 min = 6 hours duration
        self.dt_minutes = dt_minutes
        self.dt_seconds = dt_minutes * 60.0

        # Load topological graph features
        features_path = os.path.join(data_dir, "node_features.npy")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Missing {features_path}. Run src/dataset_builder.py first.")
        
        self.node_features = np.load(features_path)
        self.num_nodes = self.node_features.shape[0]

        # Extract node features: [elevation, slope, width, mannings_n, distance, cascade_prox, creep, bulking]
        self.elevations = self.node_features[:, 0]
        self.slopes = np.maximum(0.0005, self.node_features[:, 1])
        self.widths = np.maximum(10.0, self.node_features[:, 2])
        self.mannings_n = np.clip(self.node_features[:, 3], 0.02, 0.10)
        self.distances = self.node_features[:, 4] # meters
        self.bulking_factors = self.node_features[:, 7] # 1.1x to 2.0x

    def simulate_single_breach(self, Q_peak, breach_width, base_flow=50.0, time_to_peak_min=30.0):
        """
        Simulates flood wave propagation down the river reach for a given breach hydrograph.
        """
        # 1. Total bulked discharge at root (Node 0)
        bulked_Q_peak = Q_peak * self.bulking_factors[0]
        total_peak_Q = bulked_Q_peak + base_flow

        # Construct breach hydrograph at Lake (Node 0): triangular/gamma hydrograph
        time_array = np.arange(self.timesteps) * self.dt_minutes
        t_p = time_to_peak_min
        # Hydrograph: rise up to t_p, exponential attenuation after
        q_norm = np.where(time_array <= t_p, 
                          time_array / max(1.0, t_p), 
                          np.exp(-(time_array - t_p) / 45.0))
        q_root = base_flow + (bulked_Q_peak) * q_norm

        # Arrays to record node time series
        h_matrix = np.zeros((self.num_nodes, self.timesteps), dtype=np.float32)
        q_matrix = np.zeros((self.num_nodes, self.timesteps), dtype=np.float32)
        v_matrix = np.zeros((self.num_nodes, self.timesteps), dtype=np.float32)

        # Baseline depth and discharge at dry/monsoon base flow
        for i in range(self.num_nodes):
            # Manning normal depth: h = (Q * n / (W * S_0^0.5))^(3/5)
            n_i = self.mannings_n[i]
            w_i = self.widths[i]
            s_i = self.slopes[i]
            h_base = (base_flow * n_i / (w_i * np.sqrt(s_i))) ** 0.6
            h_matrix[i, :] = h_base
            q_matrix[i, :] = base_flow
            v_matrix[i, :] = base_flow / max(1.0, w_i * h_base)

        # Set root boundary condition
        q_matrix[0, :] = q_root
        for t in range(self.timesteps):
            n_0 = self.mannings_n[0]
            w_0 = self.widths[0]
            s_0 = self.slopes[0]
            h_matrix[0, t] = (q_matrix[0, t] * n_0 / (w_0 * np.sqrt(s_0))) ** 0.6
            v_matrix[0, t] = q_matrix[0, t] / max(1.0, w_0 * h_matrix[0, t])

        # 2. Propagate wave downstream via kinematic-diffusive wave routing
        arrival_times = np.zeros(self.num_nodes, dtype=np.float32)

        for i in range(1, self.num_nodes):
            dx = max(100.0, self.distances[i] - self.distances[i - 1])
            w_i = self.widths[i]
            s_i = self.slopes[i]
            n_i = self.mannings_n[i]
            bulk_ratio = self.bulking_factors[i] / max(1.0, self.bulking_factors[i - 1])

            # Wave celerity c = 5/3 * v (steep gorges have higher velocity 3.5 - 6.5 m/s)
            avg_v = max(2.8, np.mean(v_matrix[i - 1, :]) * 1.3)
            celerity = 1.67 * avg_v # m/s
            time_lag_steps = max(1, int(round((dx / celerity) / self.dt_seconds)))

            # Wave diffusion and peak attenuation along the gorge
            attenuation_factor = np.exp(-0.000025 * dx * (1.0 / max(0.001, s_i)))

            # Routed discharge with lag and diffusion
            for t in range(self.timesteps):
                if t >= time_lag_steps:
                    upstream_q = q_matrix[i - 1, t - time_lag_steps]
                    routed_q = base_flow + (upstream_q - base_flow) * attenuation_factor * bulk_ratio
                else:
                    routed_q = base_flow
                q_matrix[i, t] = max(base_flow, routed_q)

                # Manning depth
                h_i = (q_matrix[i, t] * n_i / (w_i * np.sqrt(s_i))) ** 0.6
                h_matrix[i, t] = h_i
                v_matrix[i, t] = q_matrix[i, t] / max(1.0, w_i * h_i)

            # Compute arrival time (in minutes): front threshold 12% rise over baseline
            base_d = h_matrix[i, 0]
            surge_indices = np.where(h_matrix[i, :] >= base_d + max(0.12, 0.12 * base_d))[0]
            if len(surge_indices) > 0:
                arrival_times[i] = surge_indices[0] * self.dt_minutes
            else:
                arrival_times[i] = (self.distances[i] / celerity) / 60.0

        h_max = np.max(h_matrix, axis=1)
        v_peak = np.max(v_matrix, axis=1)

        return h_matrix, arrival_times, h_max, v_peak

    def generate_synthetic_dataset(self, num_scenarios=250, output_dir="data/processed"):
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Simulating {num_scenarios} synthetic GLOF breach iterations with hydrodynamic routing...")

        np.random.seed(1337)
        # Parameter distributions
        breach_widths = np.random.uniform(30.0, 200.0, num_scenarios)      # 30m to 200m
        peak_discharges = np.random.uniform(500.0, 10000.0, num_scenarios) # 500 to 10,000 m3/s
        is_monsoon = np.random.choice([0, 1], size=num_scenarios, p=[0.35, 0.65])
        base_flows = np.where(is_monsoon == 1, 
                              np.random.uniform(150.0, 350.0, num_scenarios), 
                              np.random.uniform(25.0, 70.0, num_scenarios))

        all_h_matrices = np.zeros((num_scenarios, self.num_nodes, self.timesteps), dtype=np.float32)
        all_arrival_times = np.zeros((num_scenarios, self.num_nodes), dtype=np.float32)
        all_max_depths = np.zeros((num_scenarios, self.num_nodes), dtype=np.float32)
        all_peak_velocities = np.zeros((num_scenarios, self.num_nodes), dtype=np.float32)

        for s in range(num_scenarios):
            h_mat, arr_t, h_mx, v_pk = self.simulate_single_breach(
                Q_peak=peak_discharges[s],
                breach_width=breach_widths[s],
                base_flow=base_flows[s],
                time_to_peak_min=np.random.uniform(20.0, 45.0)
            )
            all_h_matrices[s] = h_mat
            all_arrival_times[s] = arr_t
            all_max_depths[s] = h_mx
            all_peak_velocities[s] = v_pk

        # Save synthetic ground truth arrays
        np.save(os.path.join(output_dir, "synthetic_h_matrices.npy"), all_h_matrices)
        np.save(os.path.join(output_dir, "synthetic_arrival_times.npy"), all_arrival_times)
        np.save(os.path.join(output_dir, "synthetic_max_depths.npy"), all_max_depths)
        np.save(os.path.join(output_dir, "synthetic_peak_velocities.npy"), all_peak_velocities)

        # Scenario metadata dataframe
        scenario_df = pd.DataFrame({
            "scenario_id": np.arange(num_scenarios),
            "peak_discharge_m3s": peak_discharges,
            "breach_width_m": breach_widths,
            "is_monsoon": is_monsoon,
            "base_flow_m3s": base_flows,
            "mean_arrival_time_min": np.mean(all_arrival_times, axis=1),
            "peak_water_depth_at_outlet_m": all_max_depths[:, -1]
        })
        scenario_df.to_csv(os.path.join(output_dir, "scenario_metadata.csv"), index=False)

        print(f"[+] Finished generating {num_scenarios} scenarios:")
        print(f"    - Depth matrices: {all_h_matrices.shape}")
        print(f"    - Arrival times:  {all_arrival_times.shape}")
        print(f"    - Max depths:     {all_max_depths.shape}")
        print(f"    - Peak velocities:{all_peak_velocities.shape}")
        return scenario_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Ground Truth Generator for GLOF Routing")
    parser.add_argument("--scenarios", type=int, default=200, help="Number of Monte Carlo scenarios")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save synthetic labels")
    args = parser.parse_args()

    simulator = HydrodynamicGLOFSimulator(data_dir=args.output_dir)
    simulator.generate_synthetic_dataset(num_scenarios=args.scenarios, output_dir=args.output_dir)
