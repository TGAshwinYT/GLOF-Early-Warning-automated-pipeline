"""
Kaggle Cloud Training Runner & Dataset Synchronization Agent.

Automates:
1. Dataset packaging and metadata generation for data/processed/
2. Kaggle dataset push: `kaggle datasets create -p data/processed/` (or version update)
3. Kaggle kernel push: `kaggle kernels push -p notebooks/`
4. Remote GPU training orchestration (Dual NVIDIA T4 GPUs)
"""

import os
import json
import argparse
import subprocess
import shutil

class KaggleRunner:
    def __init__(self, data_dir="data/processed", notebook_dir="notebooks"):
        self.data_dir = data_dir
        self.notebook_dir = notebook_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(notebook_dir, exist_ok=True)

    def prepare_dataset_metadata(self, title="Himalayan GLOF River Graph Dataset", dataset_slug="glof-river-graph-data"):
        metadata_path = os.path.join(self.data_dir, "dataset-metadata.json")
        username = os.environ.get("KAGGLE_USERNAME", "antigravity-agent")
        
        metadata = {
            "title": title,
            "id": f"{username}/{dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[+] Prepared dataset metadata at {metadata_path}")
        return metadata_path

    def prepare_kernel_metadata(self, kernel_slug="glof-stgnn-trainer", dataset_slug="glof-river-graph-data"):
        metadata_path = os.path.join(self.notebook_dir, "kernel-metadata.json")
        username = os.environ.get("KAGGLE_USERNAME", "antigravity-agent")

        metadata = {
            "id": f"{username}/{kernel_slug}",
            "title": "Cascading GLOF Spatio-Temporal GNN High-GPU Trainer",
            "code_file": "kaggle_trainer.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_internet": "true",
            "dataset_sources": [f"{username}/{dataset_slug}"]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[+] Prepared kernel metadata at {metadata_path}")
        return metadata_path

    def check_kaggle_auth(self):
        """
        Verifies existence of Kaggle credentials in ~/.kaggle/kaggle.json or environment variables.
        """
        kaggle_json_env = os.environ.get("KAGGLE_CONFIG_DIR")
        user_home_kaggle = os.path.expanduser("~/.kaggle/kaggle.json")
        has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

        if os.path.exists(user_home_kaggle) or has_env:
            return True, "Kaggle credentials verified."
        return False, f"Missing {user_home_kaggle} or KAGGLE_USERNAME/KAGGLE_KEY env variables."

    def push_dataset(self):
        auth_ok, msg = self.check_kaggle_auth()
        if not auth_ok:
            print(f"[!] Warning: {msg}")
            print("[i] To push to Kaggle, place your kaggle.json in ~/.kaggle/ or set environment variables.")
            return False

        self.prepare_dataset_metadata()
        cmd = ["kaggle", "datasets", "create", "-p", self.data_dir, "-u"]
        print(f"[*] Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout or res.stderr)
        return res.returncode == 0

    def push_kernel(self):
        auth_ok, msg = self.check_kaggle_auth()
        if not auth_ok:
            print(f"[!] Warning: {msg}")
            return False

        self.prepare_kernel_metadata()
        cmd = ["kaggle", "kernels", "push", "-p", self.notebook_dir]
        print(f"[*] Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout or res.stderr)
        return res.returncode == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Cloud Runner for GLOF River GNN")
    parser.add_argument("--prepare-only", action="store_true", default=True, help="Prepare metadata files without pushing")
    parser.add_argument("--push", action="store_true", help="Push dataset and kernel to Kaggle")
    args = parser.parse_args()

    runner = KaggleRunner()
    runner.prepare_dataset_metadata()
    runner.prepare_kernel_metadata()
    if args.push:
        runner.push_dataset()
        runner.push_kernel()
