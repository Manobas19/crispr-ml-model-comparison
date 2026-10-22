#!/usr/bin/env bash
# Ubuntu, Python 3.10+. Run inside the extracted crispr_ml_benchmark directory.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python benchmark.py --make-demo demo_activity.csv
python benchmark.py --input demo_activity.csv --out demo_results --demo-label --quick
python -m pip freeze > demo_results/installed_versions.txt

# If needed, install the venv package first:
# sudo apt update
# sudo apt install python3-venv python3-pip

# REAL DATA: copy this command separately after preparing activity.csv.
# python benchmark.py --input activity.csv --out real_results \
#   --models MeanBaseline,Ridge,SVR,RandomForest,XGBoost \
#   --test-fraction 0.2 --bootstrap 1000 --jobs 2 --seed 42
# python -m pip freeze > real_results/installed_versions.txt

# Full demonstration (larger search than --quick):
# python benchmark.py --input demo_activity.csv --out demo_full --demo-label
# Use a new output directory each run. Do not choose a split seed by test performance.
