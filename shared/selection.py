"""Deterministic rolling-validation and model-selection utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

ForecastFormulation = Literal["direct", "residual"]


@dataclass(frozen=True)
class RollingFold:
    """One expanding-window training and future validation period."""

    name: str
    train_start: str
    validation_start: str
    validation_end: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


ROLLING_FOLDS = (
    RollingFold("fold_1", "2013-03-01 00:00:00", "2015-01-01 00:00:00", "2015-07-01 00:00:00"),
    RollingFold("fold_2", "2013-03-01 00:00:00", "2015-07-01 00:00:00", "2016-01-01 00:00:00"),
    RollingFold("fold_3", "2013-03-01 00:00:00", "2016-01-01 00:00:00", "2016-08-01 00:00:00"),
)


@dataclass(frozen=True)
class LSTMConfig:
    """Serializable LSTM candidate used by the stratified random search."""

    candidate_id: str
    formulation: ForecastFormulation
    units: int
    dense_units: int
    dropout: float
    learning_rate: float
    l2_strength: float
    num_layers: int
    huber_delta: float
    source: str = "random"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_lstm_candidates(seed: int = 42, count: int = 16) -> list[LSTMConfig]:
    """Return two anchors, six direct samples, and eight residual samples."""

    if count != 16:
        raise ValueError("The full distinction search is defined for exactly 16 candidates.")
    candidates = [
        LSTMConfig("candidate_00", "direct", 64, 32, 0.2, 1e-3, 0.0, 1, 10.0, "basic_anchor"),
        LSTMConfig(
            "candidate_01",
            "direct",
            128,
            64,
            0.2,
            0.001074881542656793,
            1e-4,
            1,
            10.0,
            "previous_best_anchor",
        ),
    ]
    rng = np.random.default_rng(seed)
    seen = {
        (
            item.formulation,
            item.units,
            item.dense_units,
            item.dropout,
            round(item.learning_rate, 12),
            item.l2_strength,
            item.num_layers,
            item.huber_delta,
        )
        for item in candidates
    }
    formulations = ["direct"] * 6 + ["residual"] * 8
    for formulation in formulations:
        while True:
            values = (
                formulation,
                int(rng.choice([64, 128])),
                int(rng.choice([32, 64])),
                float(rng.choice([0.1, 0.2, 0.3])),
                float(np.exp(rng.uniform(np.log(3e-4), np.log(1.5e-3)))),
                float(rng.choice([0.0, 1e-5, 1e-4])),
                int(rng.choice([1, 2])),
                float(rng.choice([5.0, 10.0, 20.0])),
            )
            key = (*values[:4], round(values[4], 12), *values[5:])
            if key not in seen:
                seen.add(key)
                break
        candidates.append(LSTMConfig(f"candidate_{len(candidates):02d}", *values))
    return candidates


def aggregate_validation_results(results: pd.DataFrame) -> pd.DataFrame:
    """Average seeds within folds, then estimate uncertainty across folds."""

    required = {
        "candidate_id",
        "fold",
        "seed",
        "mae",
        "best_epoch",
        "parameters",
        "num_layers",
        "formulation",
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Validation results lack columns: {sorted(missing)}")
    fold_results = (
        results.groupby(["candidate_id", "fold"], observed=True)
        .agg(
            fold_mean_mae=("mae", "mean"),
            fold_seed_std_mae=("mae", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    grouped = (
        fold_results.groupby("candidate_id", observed=True)
        .agg(
            mean_mae=("fold_mean_mae", "mean"),
            std_mae=("fold_mean_mae", "std"),
            folds=("fold", "nunique"),
            mean_seed_std_mae=("fold_seed_std_mae", "mean"),
            min_seeds_per_fold=("seeds", "min"),
            max_seeds_per_fold=("seeds", "max"),
        )
        .reset_index()
    )
    fit_summary = (
        results.groupby("candidate_id", observed=True)
        .agg(
            evaluations=("mae", "size"),
            mean_best_epoch=("best_epoch", "mean"),
            median_best_epoch=("best_epoch", "median"),
            parameters=("parameters", "first"),
            num_layers=("num_layers", "first"),
            formulation=("formulation", "first"),
        )
        .reset_index()
    )
    grouped = grouped.merge(fit_summary, on="candidate_id", validate="one_to_one")
    grouped["mean_seed_std_mae"] = grouped["mean_seed_std_mae"].fillna(0.0)
    grouped["se_mae"] = grouped["std_mae"].fillna(0.0) / np.sqrt(grouped["folds"])
    return grouped.sort_values(["mean_mae", "parameters", "num_layers", "formulation"]).reset_index(drop=True)


def validate_search_completeness(
    results: pd.DataFrame,
    candidate_ids: Iterable[str],
    fold_names: Iterable[str],
    seeds: Iterable[int],
    *,
    source_fingerprint: str | None = None,
) -> pd.DataFrame:
    """Return the exact intended search rows or reject missing/duplicate/extra fits."""

    required = {"candidate_id", "fold", "seed"}
    missing_columns = required - set(results.columns)
    if missing_columns:
        raise ValueError(f"Search results lack columns: {sorted(missing_columns)}")
    filtered = results.copy()
    if source_fingerprint is not None:
        if "source_fingerprint" not in filtered:
            raise ValueError("Search results lack a methodology source fingerprint.")
        filtered = filtered.loc[filtered["source_fingerprint"].eq(source_fingerprint)].copy()
    candidates = tuple(candidate_ids)
    folds = tuple(fold_names)
    seed_values = tuple(int(seed) for seed in seeds)
    expected = {
        (candidate, fold, seed)
        for candidate in candidates
        for fold in folds
        for seed in seed_values
    }
    observed_rows = [
        (str(row.candidate_id), str(row.fold), int(row.seed))
        for row in filtered.itertuples(index=False)
    ]
    observed = set(observed_rows)
    if observed != expected or len(observed_rows) != len(expected):
        raise ValueError(
            f"Search completeness check failed: expected {len(expected)} unique fits, "
            f"observed {len(observed)} unique fits across {len(observed_rows)} rows."
        )
    return filtered.sort_values(["candidate_id", "fold", "seed"], kind="stable").reset_index(drop=True)


def selection_audit(summary: pd.DataFrame, selected: pd.Series) -> dict[str, Any]:
    """Return the threshold, eligible candidates, and deterministic tie rationale."""

    best = summary.sort_values("mean_mae", kind="stable").iloc[0]
    threshold = float(best["mean_mae"] + best["se_mae"])
    eligible = summary.loc[summary["mean_mae"] <= threshold, "candidate_id"].astype(str).tolist()
    return {
        "best_candidate_by_mean_mae": str(best["candidate_id"]),
        "best_mean_mae": float(best["mean_mae"]),
        "best_fold_level_standard_error": float(best["se_mae"]),
        "one_standard_error_threshold": threshold,
        "eligible_candidates": eligible,
        "selected_candidate": str(selected["candidate_id"]),
        "tie_break_reason": (
            "Among candidates within one fold-level standard error, prefer fewer parameters, "
            "then fewer LSTM layers, then direct forecasting."
        ),
    }


def select_one_standard_error(
    summary: pd.DataFrame,
    *,
    simplicity_columns: tuple[str, ...],
    preference_columns: tuple[str, ...] = (),
) -> pd.Series:
    """Apply the one-standard-error rule with deterministic simplicity ties."""

    if summary.empty:
        raise ValueError("Cannot select from an empty result table.")
    best = summary.sort_values("mean_mae").iloc[0]
    threshold = float(best["mean_mae"] + best.get("se_mae", 0.0))
    eligible = summary.loc[summary["mean_mae"] <= threshold].copy()
    sort_columns = [*simplicity_columns, *preference_columns, "mean_mae"]
    return eligible.sort_values(sort_columns, kind="stable").iloc[0]


def select_lstm_candidate(summary: pd.DataFrame) -> pd.Series:
    """Select within one SE, preferring parameters, layers, then direct form."""

    ranked = summary.copy()
    ranked["formulation_rank"] = ranked["formulation"].map({"direct": 0, "residual": 1})
    return select_one_standard_error(
        ranked,
        simplicity_columns=("parameters", "num_layers", "formulation_rank"),
    )
