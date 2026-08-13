"""Deterministic rolling-validation and model-selection utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd

ForecastFormulation = Literal["direct", "residual"]
LossName = Literal["huber", "mae", "pinball"]
OptimizerName = Literal["adam", "adamw"]


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
    """Complete serializable architecture and optimization recipe."""

    candidate_id: str
    formulation: ForecastFormulation
    units: int
    dense_units: int
    dropout: float
    learning_rate: float
    l2_strength: float
    num_layers: int
    huber_delta: float
    loss_name: LossName = "huber"
    pinball_tau: float = 0.5
    optimizer_name: OptimizerName = "adam"
    weight_decay: float = 0.0
    clipnorm: float | None = None
    use_ema: bool = False
    ema_momentum: float = 0.99
    batch_size: int = 256
    source: str = "focused_local_search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_optimization_recipes() -> list[LSTMConfig]:
    """Return the six preregistered candidate-13 optimization recipes."""

    common = {
        "formulation": "residual",
        "units": 128,
        "dense_units": 64,
        "dropout": 0.2,
        "learning_rate": 0.000559702206464,
        "l2_strength": 1e-4,
        "num_layers": 2,
        "batch_size": 256,
        "ema_momentum": 0.99,
    }
    return [
        LSTMConfig(
            "recipe_00",
            huber_delta=5.0,
            source="candidate13_reproduction",
            **common,
        ),
        LSTMConfig(
            "recipe_01",
            huber_delta=5.0,
            clipnorm=1.0,
            use_ema=True,
            source="adam_huber5_ema",
            **common,
        ),
        LSTMConfig(
            "recipe_02",
            huber_delta=2.0,
            clipnorm=1.0,
            use_ema=True,
            source="adam_huber2_ema",
            **common,
        ),
        LSTMConfig(
            "recipe_03",
            huber_delta=2.0,
            loss_name="mae",
            clipnorm=1.0,
            use_ema=True,
            source="adam_mae_ema",
            **common,
        ),
        LSTMConfig(
            "recipe_04",
            huber_delta=2.0,
            loss_name="pinball",
            pinball_tau=0.55,
            clipnorm=1.0,
            use_ema=True,
            source="adam_pinball055_ema",
            **common,
        ),
        LSTMConfig(
            "recipe_05",
            huber_delta=2.0,
            optimizer_name="adamw",
            weight_decay=1e-5,
            clipnorm=1.0,
            use_ema=True,
            source="adamw_huber2_ema",
            **common,
        ),
    ]


def _apply_recipe(
    recipe: LSTMConfig,
    *,
    candidate_id: str,
    formulation: ForecastFormulation,
    units: int,
    dense_units: int,
    dropout: float,
    learning_rate: float,
    l2_strength: float,
    num_layers: int,
    batch_size: int,
    source: str,
) -> LSTMConfig:
    return replace(
        recipe,
        candidate_id=candidate_id,
        formulation=formulation,
        units=units,
        dense_units=dense_units,
        dropout=dropout,
        learning_rate=learning_rate,
        l2_strength=l2_strength,
        num_layers=num_layers,
        batch_size=batch_size,
        source=source,
    )


def generate_lstm_candidates(
    seed: int = 42,
    count: int = 12,
    recipe: LSTMConfig | None = None,
) -> list[LSTMConfig]:
    """Return two anchors and a balanced ten-candidate local search."""

    if count != 12:
        raise ValueError("The V2.2 focused search is defined for exactly 12 candidates.")
    selected_recipe = recipe or generate_optimization_recipes()[0]
    candidates = [
        _apply_recipe(
            selected_recipe,
            candidate_id="candidate_00",
            formulation="direct",
            units=64,
            dense_units=32,
            dropout=0.2,
            learning_rate=1e-3,
            l2_strength=0.0,
            num_layers=1,
            batch_size=256,
            source="basic_anchor",
        ),
        _apply_recipe(
            selected_recipe,
            candidate_id="candidate_01",
            formulation="residual",
            units=128,
            dense_units=64,
            dropout=0.2,
            learning_rate=0.000559702206464,
            l2_strength=1e-4,
            num_layers=2,
            batch_size=256,
            source="candidate13_anchor",
        ),
    ]
    rng = np.random.default_rng(seed)
    formulations = np.asarray(["residual"] * 8 + ["direct"] * 2, dtype=object)
    rng.shuffle(formulations)

    def balanced(values: list[Any]) -> np.ndarray:
        assigned = np.resize(np.asarray(values, dtype=object), 10)
        rng.shuffle(assigned)
        return assigned

    units = balanced([64, 96, 128])
    dense_units = balanced([32, 64])
    dropouts = balanced([0.1, 0.2, 0.3])
    l2_values = balanced([0.0, 1e-5, 1e-4])
    layers = balanced([1, 2])
    batch_sizes = balanced([128, 256])
    log_edges = np.linspace(np.log(3e-4), np.log(9e-4), 11)
    learning_rates = np.exp(
        [rng.uniform(log_edges[index], log_edges[index + 1]) for index in range(10)]
    )
    rng.shuffle(learning_rates)
    for index in range(10):
        candidates.append(
            _apply_recipe(
                selected_recipe,
                candidate_id=f"candidate_{index + 2:02d}",
                formulation=formulations[index],
                units=int(units[index]),
                dense_units=int(dense_units[index]),
                dropout=float(dropouts[index]),
                learning_rate=float(learning_rates[index]),
                l2_strength=float(l2_values[index]),
                num_layers=int(layers[index]),
                batch_size=int(batch_sizes[index]),
                source="focused_local_search",
            )
        )
    keys = {
        (
            candidate.formulation,
            candidate.units,
            candidate.dense_units,
            candidate.dropout,
            round(candidate.learning_rate, 12),
            candidate.l2_strength,
            candidate.num_layers,
            candidate.batch_size,
        )
        for candidate in candidates
    }
    if len(keys) != len(candidates):
        raise RuntimeError("Focused candidate generation produced a duplicate configuration.")
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


def paired_comparison_table(
    results: pd.DataFrame,
    *,
    item_column: str = "candidate_id",
) -> pd.DataFrame:
    """Compare every item with the raw winner using paired fold-level errors."""

    required = {item_column, "fold", "seed", "mae"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Paired results lack columns: {sorted(missing)}")
    fold_means = (
        results.groupby([item_column, "fold"], observed=True)["mae"]
        .mean()
        .rename("fold_mean_mae")
        .reset_index()
    )
    counts = fold_means.groupby(item_column, observed=True)["fold"].nunique()
    if counts.nunique() != 1:
        raise ValueError("Paired selection requires the same folds for every item.")
    pivot = fold_means.pivot(index="fold", columns=item_column, values="fold_mean_mae")
    if pivot.isna().any().any():
        raise ValueError("Paired selection requires a complete item-by-fold result matrix.")
    mean_errors = pivot.mean(axis=0)
    raw_winner = str(mean_errors.sort_values(kind="stable").index[0])
    rows: list[dict[str, Any]] = []
    for item in pivot.columns:
        differences = pivot[item] - pivot[raw_winner]
        paired_se = float(differences.std(ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else 0.0
        mean_degradation = float(differences.mean())
        rows.append(
            {
                item_column: item,
                "raw_winner": raw_winner,
                "mean_paired_degradation": mean_degradation,
                "paired_standard_error": paired_se,
                "paired_one_se_eligible": bool(mean_degradation <= paired_se + 1e-12),
                "paired_folds": int(len(differences)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_paired_degradation", item_column], kind="stable"
    ).reset_index(drop=True)


def select_paired_one_standard_error(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    item_column: str = "candidate_id",
    simplicity_columns: tuple[str, ...],
    preference_columns: tuple[str, ...] = (),
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Select a simple model only when paired fold evidence supports equivalence."""

    comparisons = paired_comparison_table(results, item_column=item_column)
    ranked = summary.merge(comparisons, on=item_column, validate="one_to_one")
    eligible = ranked.loc[ranked["paired_one_se_eligible"]].copy()
    if eligible.empty:
        raise RuntimeError("Paired one-standard-error selection produced no eligible item.")
    sort_columns = [*simplicity_columns, *preference_columns, "mean_mae", item_column]
    selected = eligible.sort_values(sort_columns, kind="stable").iloc[0]
    raw_winner = str(comparisons.iloc[0]["raw_winner"])
    audit = {
        "raw_winner_by_mean_mae": raw_winner,
        "selected_item": str(selected[item_column]),
        "eligible_items": comparisons.loc[
            comparisons["paired_one_se_eligible"], item_column
        ].astype(str).tolist(),
        "rule": (
            "Average seeds within folds; compare each item with the raw winner; "
            "accept equivalence only when mean paired degradation is no greater "
            "than its fold-level paired standard error."
        ),
        "tie_break_columns": [*simplicity_columns, *preference_columns],
    }
    return selected, comparisons, audit


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


def select_lstm_candidate(
    results: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, dict[str, Any]]:
    """Use paired fold evidence, then parameters, layers, and direct form."""

    ranked = summary.copy()
    ranked["formulation_rank"] = ranked["formulation"].map({"direct": 0, "residual": 1})
    return select_paired_one_standard_error(
        results,
        ranked,
        item_column="candidate_id",
        simplicity_columns=("parameters", "num_layers", "formulation_rank"),
    )
