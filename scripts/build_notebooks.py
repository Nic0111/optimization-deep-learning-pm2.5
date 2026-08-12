"""Generate five local-first notebooks from readable cell definitions.

Expensive work lives in ``shared.workflow`` so the CLI and notebooks use the
same implementation.  During automated rendering, completed artifacts are
loaded and displayed; missing stages raise instead of silently retraining.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def _cell_id(source: str, index: int) -> str:
    return hashlib.sha1(f"{index}:{source}".encode()).hexdigest()[:12]


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


def write_notebook(relative_path: str, cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        cell["id"] = _cell_id(cell["source"], index)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (odl-assignment)",
                "language": "python",
                "name": "odl-assignment",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def bootstrap_cells(title: str, purpose: str, stage: str) -> list[dict]:
    return [
        markdown(
            f"""
            # {title}

            **Purpose:** {purpose}

            This notebook is an executable evidence view over the shared local pipeline. The
            command-line runner and notebook call the same stage implementation, preventing code
            drift. Completed artifacts are reused; no Google Colab or Google Drive runtime is used.
            """
        ),
        code(
            """
            # Local WSL project bootstrap. Dependency installation belongs to the
            # locked project environment and is intentionally not repeated in notebooks.
            from pathlib import Path
            import json
            import os
            import platform
            import sys

            candidate = Path.cwd().resolve()
            PROJECT_ROOT = next(
                (path for path in [candidate, *candidate.parents] if (path / "shared").exists()),
                candidate,
            )
            if not (PROJECT_ROOT / "shared").exists():
                raise FileNotFoundError(
                    "Open this notebook from the local odl-assignment project root."
                )
            os.environ["ODL_PROJECT_ROOT"] = str(PROJECT_ROOT)
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from shared.config import PATHS
            from shared.runtime import RuntimeOptions
            from shared.workflow import run_stage

            OPTIONS = RuntimeOptions.from_environment()
            print("Project root:", PROJECT_ROOT)
            print("Run mode:", OPTIONS.mode)
            print("Device policy:", OPTIONS.device)
            print("WSL distribution:", os.getenv("WSL_DISTRO_NAME", "not detected"))
            """
        ),
        code(
            f"""
            # Run the shared stage or load its verified completed result.
            STAGE_SUMMARY = run_stage("{stage}", options=OPTIONS)
            STAGE_SUMMARY
            """
        ),
        code(
            """
            # Record the exact runtime used for this output when preflight evidence exists.
            environment_path = PATHS.environment_artifacts / f"environment_{OPTIONS.mode}.json"
            ENVIRONMENT = (
                json.loads(environment_path.read_text(encoding="utf-8"))
                if environment_path.exists()
                else {"status": "preflight evidence not yet generated"}
            )
            ENVIRONMENT
            """
        ),
    ]


def build_acquisition_notebook() -> None:
    cells = bootstrap_cells(
        "Assignment 1 — Data Acquisition and Source Audit",
        "Download the official UCI archive, extract all 12 stations, and verify source integrity.",
        "acquire",
    )
    cells += [
        markdown(
            """
            ## Why this dataset and target are defensible

            **Decision.** Use the official UCI Beijing Multi-Site Air-Quality dataset to predict
            next-hour PM2.5. **Alternatives.** Smaller single-station datasets and classification
            targets were rejected because they provide less temporal, spatial, and regression
            complexity. **Evidence.** The source contains 420,768 hourly rows, 12 stations,
            meteorology, co-pollutants, missing runs, and extreme events. **Support.** The UCI DOI
            identifies an open, reproducible secondary source
            ([UCI dataset](https://doi.org/10.24432/C5RK5G)). **Trade-off.** Results describe
            historical Beijing monitoring conditions and are predictive, not causal or universally
            generalisable.
            """
        ),
        markdown(
            """
            ## Source provenance

            The source manifest records the UCI URL, DOI, licence, retrieval timestamp, archive
            size, and SHA-256 digest. This makes the downloaded input independently identifiable.
            """
        ),
        code(
            """
            import pandas as pd

            source_manifest = json.loads(PATHS.source_manifest.read_text(encoding="utf-8"))
            pd.DataFrame([source_manifest])
            """
        ),
        markdown(
            """
            ## Structural integrity

            The audit proves the expected 420,768 hourly records, 12 stations, date range, unique
            station-time keys, and uninterrupted timestamp grid before modelling begins.
            """
        ),
        code(
            """
            station_audit = pd.read_csv(PATHS.assignment1_tables / "station_source_audit.csv")
            raw_schema = pd.read_csv(PATHS.assignment1_tables / "raw_schema.csv")
            display(station_audit)
            display(raw_schema)
            """
        ),
        markdown(
            """
            ## Missing measurements before preprocessing

            Missing values are reported rather than silently filled during acquisition. Their
            treatment is performed causally in the next notebook.
            """
        ),
        code(
            """
            raw_missing = pd.read_csv(PATHS.assignment1_tables / "raw_missing_summary.csv")
            missing_by_station = pd.read_csv(
                PATHS.assignment1_tables / "raw_missing_by_station.csv"
            )
            display(raw_missing)
            display(missing_by_station)
            print("Acquisition and source audit complete.")
            """
        ),
    ]
    write_notebook("assignment1/01_data_acquisition_and_audit.ipynb", cells)


def build_eda_notebook() -> None:
    cells = bootstrap_cells(
        "Assignment 1 — Exploratory Data Analysis and Preprocessing",
        "Analyse the validated data and create leakage-safe model inputs for both architectures.",
        "prepare",
    )
    cells += [
        markdown(
            """
            ## Distribution, missingness, and station differences

            These outputs quantify the target's skew, extreme values, spatial variation, and the
            length of missing runs before causal imputation.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import Image, display

            descriptive = pd.read_csv(
                PATHS.assignment1_tables / "descriptive_statistics.csv", index_col=0
            )
            missing_runs = pd.read_csv(PATHS.assignment1_tables / "missing_run_summary.csv")
            station_statistics = pd.read_csv(
                PATHS.assignment1_tables / "pm25_station_statistics.csv"
            )
            display(descriptive)
            display(missing_runs.sort_values("maximum_run_hours", ascending=False).head(20))
            display(station_statistics)
            display(Image(filename=str(PATHS.assignment1_figures / "missingness_heatmap.png")))
            display(Image(filename=str(
                PATHS.assignment1_figures / "pm25_distribution_and_station_boxplots.png"
            )))
            """
        ),
        markdown(
            """
            ## Temporal and feature relationships

            Hourly/seasonal behaviour and autocorrelation justify cyclical features and the
            24-hour sequence length. Correlation and sampled scatter plots provide complementary
            evidence about contemporaneous predictors.
            """
        ),
        code(
            """
            display(Image(filename=str(PATHS.assignment1_figures / "pm25_temporal_patterns.png")))
            display(Image(filename=str(
                PATHS.assignment1_figures / "continuous_correlation_heatmap.png"
            )))
            display(Image(filename=str(PATHS.assignment1_figures / "pm25_autocorrelation.png")))
            display(Image(filename=str(
                PATHS.assignment1_figures / "pm25_feature_relationships.png"
            )))
            display(Image(filename=str(
                PATHS.assignment1_figures / "selected_pollution_episode.png"
            )))
            """
        ),
        markdown(
            """
            ## Leakage-safe preprocessing and chronological splits

            **Decision.** Use capped causal forward filling, training-only station/global medians,
            training-fitted scaling, and target-time chronological splits. **Alternatives.** Random
            splitting leaks future regimes; unrestricted filling can freeze a sensor for weeks;
            interpolation can use future values. **Evidence.** The missing-run table includes long
            pollutant gaps, while the ablation tables select the fill cap and lookback over three
            expanding validation folds. **Support.** Rolling evaluation respects forecasting order
            ([Bergmeir et al.](https://doi.org/10.1016/j.csda.2017.11.003)). **Trade-off.** Median
            fallback is robust but smooths unusual conditions. Targets retain original, unfilled
            measurements and splits are assigned by target timestamp.
            """
        ),
        code(
            """
            feature_metadata = json.loads(PATHS.feature_metadata.read_text(encoding="utf-8"))
            split_metadata = json.loads(PATHS.split_metadata.read_text(encoding="utf-8"))
            split_distribution = pd.read_csv(
                PATHS.assignment1_tables / "target_distribution_by_split.csv", index_col=0
            )
            suffix = OPTIONS.artifact_suffix
            fill_ablation = pd.read_csv(
                PATHS.assignment2_tables / f"forward_fill_ablation_summary{suffix}.csv"
            )
            lookback_ablation = pd.read_csv(
                PATHS.assignment2_tables / f"lookback_ablation_summary{suffix}.csv"
            )
            display(feature_metadata)
            display(split_metadata)
            display(split_distribution)
            display(fill_ablation)
            display(lookback_ablation)
            display(Image(filename=str(
                PATHS.assignment1_figures / "target_distribution_by_split.png"
            )))
            print("EDA and causal preprocessing complete.")
            """
        ),
    ]
    write_notebook("assignment1/02_eda_and_preprocessing.ipynb", cells)


def build_baselines_notebook() -> None:
    cells = bootstrap_cells(
        "Assignment 2 — Baselines and Basic Deep-Learning Models",
        "Compare persistence, LSTM, and causal 1D CNN models under one feature contract.",
        "baselines",
    )
    cells += [
        markdown(
            """
            ## Baseline forecasts

            **Decision.** Compare neural networks with persistence and matched seasonal
            persistence. **Alternatives.** Reporting neural models alone cannot show whether they
            beat the strong lag-1 autocorrelation. **Evidence.** The EDA reports lag-1 correlation
            near 0.97. **Trade-off.** Naive baselines are intentionally simple, but provide the
            minimum credible forecasting reference.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import Image, display

            suffix = OPTIONS.artifact_suffix
            baseline_metrics = pd.read_csv(
                PATHS.assignment2_tables / f"validation_baseline_metrics{suffix}.csv"
            )
            display(baseline_metrics)
            """
        ),
        markdown(
            """
            ## Controlled architecture comparison

            **Decision.** Compare an LSTM with a causal dilated 1D CNN using identical features,
            rolling folds, loss, and primary metric. LSTM is designed for gated sequential memory
            ([Hochreiter and Schmidhuber](https://doi.org/10.1162/neco.1997.9.8.1735)); the CNN is
            a parallel local-pattern alternative whose final causal state preserves recent
            information ([Bai et al.](https://arxiv.org/abs/1803.01271)). **Trade-off.** This is an
            architecture comparison, not proof that either family is universally superior.
            """
        ),
        code(
            """
            comparison = pd.read_csv(
                PATHS.assignment2_tables / f"basic_model_comparison{suffix}.csv"
            )
            selection = json.loads(
                (PATHS.models / f"selected_architecture{suffix}.json").read_text(encoding="utf-8")
            )
            display(comparison)
            display(selection)
            display(pd.read_csv(
                PATHS.assignment2_tables / f"basic_rolling_results{suffix}.csv"
            ))
            display(Image(filename=str(
                PATHS.assignment2_figures / f"basic_lstm_training{suffix}.png"
            )))
            display(Image(filename=str(
                PATHS.assignment2_figures / f"basic_cnn_training{suffix}.png"
            )))
            print("Selected architecture:", selection["selected_model"].upper())
            """
        ),
    ]
    write_notebook("assignment2/01_baselines_and_basic_models.ipynb", cells)


def build_tuning_notebook() -> None:
    cells = bootstrap_cells(
        "Assignment 2 — Hyperparameter Tuning",
        "Tune the validation-selected architecture with repeated-seed rolling random search.",
        "tune",
    )
    cells += [
        markdown(
            """
            ## Stratified random-search evidence

            **Decision.** Evaluate 16 deterministic candidates across three rolling folds and seeds
            42/52. Two anchors preserve the basic and previous-best settings; remaining candidates
            balance direct and persistence-residual formulations. Random search is efficient when
            only some dimensions strongly affect performance
            ([Bergstra and Bengio](https://www.jmlr.org/papers/v13/bergstra12a.html)). **Alternatives.**
            The previous single-execution search result was smaller than seed variation.
            **Trade-off.** Six validation fits per candidate cost more but make selection more
            defensible. The frozen test benchmark is never read during selection.
            """
        ),
        code(
            """
            import pandas as pd

            suffix = OPTIONS.artifact_suffix
            trials = pd.read_csv(
                PATHS.assignment2_tables / f"random_search_rolling_results{suffix}.csv"
            )
            summary = pd.read_csv(
                PATHS.assignment2_tables / f"random_search_summary{suffix}.csv"
            )
            best_hyperparameters = json.loads(
                (PATHS.models / f"best_hyperparameters{suffix}.json").read_text(encoding="utf-8")
            )
            display(summary)
            display(trials)
            display(best_hyperparameters)
            """
        ),
        markdown(
            """
            ## Final validation selection and fixed official model

            **Decision.** Apply the one-standard-error rule, preferring fewer parameters, fewer
            layers, then a direct formulation when performance is practically tied. The selected
            configuration is trained once with seed 42 on all pre-test data for the median
            validation-derived epoch budget. Huber loss limits the influence of extreme spikes
            ([Huber](https://doi.org/10.1214/aoms/1177703732)). **Trade-off.** Simplicity may give up
            a tiny mean-MAE advantage in exchange for stability and clearer attribution.
            """
        ),
        code(
            """
            final_selection = json.loads(
                (PATHS.models / f"final_model_selection{suffix}.json").read_text(encoding="utf-8")
            )
            display(final_selection)
            print("Rolling tuning and single-model selection complete.")
            """
        ),
    ]
    write_notebook("assignment2/02_hyperparameter_tuning.ipynb", cells)


def build_evaluation_notebook() -> None:
    cells = bootstrap_cells(
        "Assignment 2 — Final Test Evaluation",
        "Evaluate frozen models once on the previously inspected chronological benchmark.",
        "evaluate",
    )
    cells += [
        markdown(
            """
            ## Overall comparison and computational evidence

            **Decision.** MAE is primary because it is interpretable in µg/m³ and less dominated by
            spikes; RMSE exposes large misses, R² measures explained variation, and sMAPE provides
            relative error. Parameter and timing evidence adds efficiency. **Trade-off.** No single
            metric captures public-health costs, so all claims use the metric panel and subgroup
            evidence. This period is called a frozen benchmark—not untouched—because aggregate
            results from the earlier methodology were already inspected.
            """
        ),
        code(
            """
            import pandas as pd
            from IPython.display import Image, display

            suffix = OPTIONS.artifact_suffix
            overall_metrics = pd.read_csv(
                PATHS.assignment2_tables / f"final_overall_metrics{suffix}.csv"
            )
            inference = pd.read_csv(
                PATHS.assignment2_tables / f"inference_timing{suffix}.csv"
            )
            display(overall_metrics)
            display(inference)
            """
        ),
        markdown(
            """
            ## Robustness by station, season, and pollution intensity

            Grouped metrics expose where aggregate performance hides difficult conditions. Seed
            tables compare every model rather than only the winner.
            """
        ),
        code(
            """
            station_metrics = pd.read_csv(
                PATHS.assignment2_tables / f"final_metrics_by_station{suffix}.csv"
            )
            season_metrics = pd.read_csv(
                PATHS.assignment2_tables / f"final_metrics_by_season{suffix}.csv"
            )
            pollution_metrics = pd.read_csv(
                PATHS.assignment2_tables / f"final_metrics_by_pollution_band{suffix}.csv"
            )
            residual_diagnostics = pd.read_csv(
                PATHS.assignment2_tables / f"residual_diagnostics{suffix}.csv"
            )
            display(station_metrics)
            display(season_metrics)
            display(pollution_metrics)
            display(residual_diagnostics)
            """
        ),
        markdown(
            """
            ## Temporal uncertainty and diagnostic plots

            Paired station-day block bootstrap intervals preserve more temporal dependence than
            resampling independent hours. Diagnostic plots reveal bias, extremes, and time-local
            behaviour.
            """
        ),
        code(
            """
            bootstrap = pd.read_csv(
                PATHS.assignment2_tables / f"paired_block_bootstrap{suffix}.csv"
            )
            display(bootstrap)
            optimization_claim = json.loads(
                (PATHS.assignment2_tables / f"optimization_claim{suffix}.json").read_text(
                    encoding="utf-8"
                )
            )
            display(optimization_claim)
            display(Image(filename=str(
                PATHS.assignment2_figures / f"final_actual_vs_predicted{suffix}.png"
            )))
            display(Image(filename=str(
                PATHS.assignment2_figures / f"final_station_mae{suffix}.png"
            )))
            display(Image(filename=str(
                PATHS.assignment2_figures / f"final_example_week{suffix}.png"
            )))
            display(Image(filename=str(
                PATHS.assignment2_figures / f"final_residual_analysis{suffix}.png"
            )))
            print(
                "Published metrics are not directly comparable when datasets, horizons, splits, "
                "or preprocessing differ. Final benchmark evaluation complete."
            )
            """
        ),
    ]
    write_notebook("assignment2/03_final_evaluation.ipynb", cells)


def main() -> None:
    build_acquisition_notebook()
    build_eda_notebook()
    build_baselines_notebook()
    build_tuning_notebook()
    build_evaluation_notebook()
    print("Generated five local-first notebooks under assignment1/ and assignment2/.")


if __name__ == "__main__":
    main()
