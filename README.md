# Beijing Multi-Site Air-Quality Forecasting

This submission contains the V2.2 implementation and evidence for one-hour-ahead
PM2.5 forecasting across 12 Beijing monitoring stations. It compares persistence
baselines, a basic LSTM, a causal 1D CNN and a validation-selected tuned LSTM.

## Submission contents

- `assignment1/` and `assignment2/`: five executed assessment notebooks.
- `shared/`: preprocessing, sequence construction, models, selection and evaluation.
- `scripts/`: WSL setup and resumable experiment runner.
- `tests/`: automated leakage, selection and reproducibility checks.
- `data/raw/`: original UCI station CSV files and source-verification manifest.
- `outputs/`: final tables and figures used by the notebooks and report.
- `artifacts/models/`: saved Keras models and final-selection audit files.
- `artifacts/histories/`: saved training histories.
- `artifacts/preprocessing/`: fitted preprocessing objects used with the saved models.
- `pyproject.toml` and `uv.lock`: locked Python dependency specification.

Generated intermediate and processed data are intentionally omitted because the
pipeline recreates them from `data/raw/`.

## Environment

The project was run in Ubuntu 24.04 under WSL2 with Python 3.12 and TensorFlow
2.21.0. An NVIDIA CUDA-capable GPU is recommended for the full experiment. From
the project root inside WSL, create the locked environment with:

```bash
bash scripts/bootstrap_wsl.sh
```

The bootstrap installs `uv` 0.11.3 when needed, creates `.venv`, installs the
locked dependencies and registers the `odl-assignment` Jupyter kernel.

## Verification and reproduction

Run the automated tests:

```bash
uv run pytest
```

Run the complete smoke experiment first:

```bash
./scripts/odl.sh run --mode smoke --device gpu --batch-size 256 --render
```

After the smoke run succeeds, reproduce the complete experiment and execute all
five notebooks:

```bash
./scripts/odl.sh run --mode full --device gpu --batch-size 256 --render
```

The pipeline is resumable by default. Its status can be inspected with:

```bash
./scripts/odl.sh status --mode full
```

The full experiment is computationally expensive and may require many hours on a
laptop GPU. Different compatible GPU and CUDA implementations may produce small
floating-point differences despite the configured seeds and deterministic controls.

## Key saved evidence

- Final metrics: `outputs/assignment2/tables/final_overall_metrics.csv`
- Final model-selection audit: `artifacts/models/final_model_selection.json`
- Validation gate: `artifacts/models/focused_lstm_validation_gate.json`
- Saved trained models: `artifacts/models/*.keras`
- Training histories: `artifacts/histories/`

The primary model-selection metric is MAE. The tuned LSTM achieved the lowest
observed final-benchmark MAE, while its improvement over the basic LSTM was not
statistically established by the official seven-day block-bootstrap interval.

## Dataset provenance

The raw files are from the UCI Beijing Multi-Site Air Quality dataset
(DOI: `10.24432/C5RK5G`). `data/raw/source_manifest.json` records the download
source and SHA-256 verification information used by the acquisition pipeline.
