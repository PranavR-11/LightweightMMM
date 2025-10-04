# Analyst MMM Pipeline → Tableau Public

This script:
1. Trains a Bayesian MMM (LightweightMMM) with Strong-style Fourier seasonality (annual + monthly).
2. Exports:
   - `predictions.csv` → actual vs predicted weekly sales
   - `scenarios.csv` → grid of allocation scenarios with predicted totals

## Run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py