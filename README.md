# Bread Forecasting for Retail Fresh

This repository supports the [Retail Fresh project](https://prosjektbanken.forskningsradet.no/en/project/FORISS/336988),
with a focus on **reducing bread waste in stores** by predicting daily sales at multiple levels.

The goal is accurate, actionable forecasts for:
- store total demand
- product share within each store-day
- direct product quantity
- hierarchical reconstruction (`pred_share * pred_total`)

## What’s Included
- Notebook workflows for EDA, data preparation, model training, and hierarchical evaluation
- Reusable feature engineering and modeling modules in `src/`
- Diagnostics and feature audit helpers in `src/analysis/`

**Data files are not included in this repo.** The `data/` directory is ignored by Git since the data is not public. So one needs to keep raw and processed
data locally.

## Models in This Repository
- **XGBoost (total, share, direct)**: main classical baselines
- **Prophet (total)**: per-store total demand forecasting with optional regressors
- **AutoGluon (time series)**: optional track for store/product experiments
- **Hierarchical evaluation**: reconstructs product quantity from total × share

## Notebook Workflow

1. **EDA and diagnostics** (optional but recommended)
   - `notebooks/00_eda.ipynb`
2. **Core data build**
   - `notebooks/01_prepare_data.ipynb`
   - Outputs to `data/processed/`:
     - `sales_daily.parquet`
     - `store_daily.parquet`
     - `share_daily.parquet`
     - `direct_daily.parquet`
3. **Model training/evaluation**
   - `notebooks/02_train_models.ipynb` (classical)
   - `notebooks/02_train_models_horizon.ipynb` (horizon variant)
   - `notebooks/02b_train_autogluon.ipynb` (AutoGluon)
4. **Hierarchical evaluation**
   - `notebooks/03_train_hierarchical.ipynb`
   - `notebooks/03_train_hierarchical_horizon.ipynb`

## Environment Setup

Use separate environments for classical models and AutoGluon.

### Quick setup (recommended)
```bash
./scripts/setup_core.sh
./scripts/setup_autogluon.sh
```

By default, these scripts create virtual environments in `~/.venvs/` (outside the repository).

AutoGluon note: use Python 3.10 or 3.11 for `setup_autogluon.sh`.
If needed, provide a specific interpreter:
```bash
PYTHON_BIN=/path/to/python3.10 ./scripts/setup_autogluon.sh
```

### Core environment (XGBoost + Prophet)
```bash
python3.10 -m venv ~/.venvs/bread-core
source ~/.venvs/bread-core/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-core.txt
python -m ipykernel install --user --name bread-core --display-name "bread-core"
```

### AutoGluon environment
```bash
python3.10 -m venv ~/.venvs/bread-autogluon
source ~/.venvs/bread-autogluon/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-autogluon.txt
python -m ipykernel install --user --name bread-autogluon --display-name "bread-autogluon"
```

## Tests
```bash
source ~/.venvs/bread-core/bin/activate
pytest
```

## Publishing Notes (Public GitHub)
Notebook outputs are visible on GitHub. To keep outputs while removing local paths,
run the scrubber before pushing:

```bash
python scripts/scrub_notebook_paths.py --path notebooks
```

## Feature Documentation
Detailed definitions live in:
- `docs/DATA_DICTIONARY.md`
- `docs/FEATURE_CATALOG.md`

## Feature-Pruning Compare
You can compare baseline vs pruned feature sets in:
- `notebooks/02_train_models.ipynb`
- `notebooks/02_train_models_horizon.ipynb`

Or via CLI:
```bash
python scripts/feature_prune_compare.py \
  --dataset data/processed/store_daily.parquet \
  --model-type xgb_total \
  --horizon-days 7 \
  --top-k 20
```

## Repository Layout
```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/                      # local data (ignored by Git)
├── docs/
│   ├── DATA_DICTIONARY.md
│   └── FEATURE_CATALOG.md
├── models/                    # local model artifacts (ignored by Git)
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_prepare_data.ipynb
│   ├── 02_train_models.ipynb
│   ├── 02_train_models_horizon.ipynb
│   ├── 02b_train_autogluon.ipynb
│   ├── 03_train_hierarchical.ipynb
│   └── 03_train_hierarchical_horizon.ipynb
├── results/                   # local reports/metrics (ignored by Git)
├── scripts/
│   ├── audit_horizon_features.py
│   ├── cleanup_notebook_artifacts.sh
│   ├── feature_prune_compare.py
│   ├── scrub_notebook_paths.py
│   ├── setup_autogluon.sh
│   ├── setup_core.sh
│   └── verify_repo_hygiene.sh
├── src/
│   ├── analysis/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   ├── utils/
│   └── total.py
├── tests/
│   ├── conftest.py
│   ├── test_encoding.py
│   ├── test_leakage.py
│   ├── test_preprocessing_contract.py
│   └── test_prophet_keys.py
├── .gitignore
├── pytest.ini
├── requirements.txt
├── requirements-core.txt
├── requirements-autogluon.txt
└── README.md
```
