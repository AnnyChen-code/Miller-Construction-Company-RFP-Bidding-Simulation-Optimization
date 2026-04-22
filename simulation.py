from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRIANGULAR_SYNONYMS = {
    "item": ["item", "task", "cost_item", "name", "activity", "category"],
    "low": ["low", "min", "minimum", "optimistic", "best_case"],
    "mode": ["mode", "most_likely", "ml", "base", "likely"],
    "high": ["high", "max", "maximum", "pessimistic", "worst_case"],
    "quantity": ["quantity", "qty", "units", "count"],
}


@dataclass(frozen=True)
class CompetitorProfile:
    name: str
    low_bid: float
    mode_bid: float
    high_bid: float
    participation_probability: float = 1.0


def _normalize_name(value: str) -> str:
    return value.strip().lower().replace(" ", "_")


def _find_column(columns: Iterable[str], aliases: list[str]) -> str | None:
    normalized_to_original = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        if alias in normalized_to_original:
            return normalized_to_original[alias]
    return None


def normalize_cost_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    item_col = _find_column(df.columns, TRIANGULAR_SYNONYMS["item"])
    low_col = _find_column(df.columns, TRIANGULAR_SYNONYMS["low"])
    mode_col = _find_column(df.columns, TRIANGULAR_SYNONYMS["mode"])
    high_col = _find_column(df.columns, TRIANGULAR_SYNONYMS["high"])
    quantity_col = _find_column(df.columns, TRIANGULAR_SYNONYMS["quantity"])

    missing = [
        friendly
        for friendly, col in {
            "low": low_col,
            "mode": mode_col,
            "high": high_col,
        }.items()
        if col is None
    ]
    if missing:
        raise ValueError(
            "Missing required columns for triangular inputs: "
            + ", ".join(missing)
            + ". Expected columns like low/mode/high or min/most_likely/max."
        )

    normalized = pd.DataFrame(
        {
            "item": df[item_col] if item_col else [f"Cost Item {i + 1}" for i in range(len(df))],
            "low": pd.to_numeric(df[low_col], errors="coerce"),
            "mode": pd.to_numeric(df[mode_col], errors="coerce"),
            "high": pd.to_numeric(df[high_col], errors="coerce"),
            "quantity": 1.0 if quantity_col is None else pd.to_numeric(df[quantity_col], errors="coerce"),
        }
    )

    normalized["item"] = normalized["item"].fillna("Unnamed Item").astype(str)
    normalized["quantity"] = normalized["quantity"].fillna(1.0)
    return normalized


def validate_cost_dataframe(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The cost table is empty.")

    numeric_cols = ["low", "mode", "high", "quantity"]
    if df[numeric_cols].isna().any().any():
        raise ValueError("Some triangular inputs or quantities are missing or non-numeric.")

    if (df["quantity"] <= 0).any():
        raise ValueError("All quantities must be greater than 0.")

    invalid = ~(df["low"] <= df["mode"]) | ~(df["mode"] <= df["high"])
    if invalid.any():
        bad_items = ", ".join(df.loc[invalid, "item"].astype(str).tolist())
        raise ValueError(f"Each item must satisfy low <= mode <= high. Fix these rows: {bad_items}")

    if (df[["low", "mode", "high"]] < 0).any().any():
        raise ValueError("Cost inputs must be non-negative.")


def validate_competitors(competitors: list[CompetitorProfile]) -> None:
    if not competitors:
        raise ValueError("Add at least one competitor profile.")
    for competitor in competitors:
        if not (competitor.low_bid <= competitor.mode_bid <= competitor.high_bid):
            raise ValueError(
                f"{competitor.name} must satisfy low <= mode <= high for competitor bids."
            )
        if not (0.0 <= competitor.participation_probability <= 1.0):
            raise ValueError(
                f"{competitor.name} participation probability must be between 0 and 1."
            )


def triangular_mean(low: float, mode: float, high: float) -> float:
    return (low + mode + high) / 3.0


def triangular_std(low: float, mode: float, high: float) -> float:
    variance = (
        low * low
        + mode * mode
        + high * high
        - low * mode
        - low * high
        - mode * high
    ) / 18.0
    return float(np.sqrt(variance))


def summarize_cost_inputs(cost_df: pd.DataFrame) -> pd.DataFrame:
    df = cost_df.copy()
    df["expected_unit_cost"] = df.apply(
        lambda row: triangular_mean(row["low"], row["mode"], row["high"]), axis=1
    )
    df["stdev_unit_cost"] = df.apply(
        lambda row: triangular_std(row["low"], row["mode"], row["high"]), axis=1
    )
    df["expected_total_cost"] = df["expected_unit_cost"] * df["quantity"]
    return df


def simulate_total_project_cost(
    cost_df: pd.DataFrame,
    simulations: int,
    random_seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    total_cost = np.zeros(simulations, dtype=float)
    for row in cost_df.itertuples(index=False):
        draws = rng.triangular(row.low, row.mode, row.high, size=simulations)
        total_cost += draws * row.quantity
    return total_cost


def expected_total_cost(cost_df: pd.DataFrame) -> float:
    summary = summarize_cost_inputs(cost_df)
    return float(summary["expected_total_cost"].sum())


def simulate_competitor_bids(
    competitors: list[CompetitorProfile],
    simulations: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    bids: dict[str, np.ndarray] = {}
    for competitor in competitors:
        active = rng.random(simulations) <= competitor.participation_probability
        draw = rng.triangular(
            competitor.low_bid,
            competitor.mode_bid,
            competitor.high_bid,
            size=simulations,
        )
        bids[competitor.name] = np.where(active, draw, np.inf)
    return pd.DataFrame(bids)


def evaluate_bid_grid(
    cost_df: pd.DataFrame,
    competitors: list[CompetitorProfile],
    bid_grid: list[float],
    simulations: int,
    random_seed: int,
    fixed_overhead: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_cost_dataframe(cost_df)
    validate_competitors(competitors)
    if not bid_grid:
        raise ValueError("Provide at least one Miller bid amount.")

    project_costs = simulate_total_project_cost(
        cost_df=cost_df,
        simulations=simulations,
        random_seed=random_seed,
    )
    competitor_bids = simulate_competitor_bids(
        competitors=competitors,
        simulations=simulations,
        random_seed=random_seed + 1,
    )
    competing_min = competitor_bids.min(axis=1).to_numpy()
    no_competition = np.isinf(competing_min)

    rows = []
    scenario_details = []
    expected_cost_with_overhead = expected_total_cost(cost_df) + fixed_overhead
    for bid_amount in bid_grid:
        wins = np.logical_or(no_competition, bid_amount < competing_min)
        profit_if_won = bid_amount - (project_costs + fixed_overhead)
        realized_profit = np.where(wins, profit_if_won, 0.0)
        implied_markup_pct = ((bid_amount / expected_cost_with_overhead) - 1.0) * 100.0

        rows.append(
            {
                "bid_amount": bid_amount,
                "implied_markup_pct": implied_markup_pct,
                "win_probability": float(wins.mean()),
                "expected_profit": float(realized_profit.mean()),
                "expected_profit_if_won": float(profit_if_won[wins].mean()) if wins.any() else np.nan,
                "loss_probability_if_won": float((profit_if_won[wins] < 0).mean()) if wins.any() else np.nan,
                "worst_case_profit_if_won": float(profit_if_won[wins].min()) if wins.any() else np.nan,
                "value_at_risk_5pct": float(np.quantile(realized_profit, 0.05)),
            }
        )

        scenario_details.append(
            pd.DataFrame(
                {
                    "bid_amount": bid_amount,
                    "won_project": wins,
                    "project_cost": project_costs,
                    "lowest_competitor_bid": competing_min,
                    "realized_profit": realized_profit,
                }
            )
        )

    results = pd.DataFrame(rows).sort_values("expected_profit", ascending=False).reset_index(drop=True)
    details = pd.concat(scenario_details, ignore_index=True)
    return results, details


def run_diagnostics() -> pd.DataFrame:
    sample = pd.DataFrame(
        {
            "item": ["Earthwork", "Concrete", "Steel"],
            "low": [100, 200, 300],
            "mode": [120, 240, 360],
            "high": [150, 300, 450],
            "quantity": [1, 2, 1],
        }
    )
    competitors = [
        CompetitorProfile("Known competitor", 500, 700, 900, 1.0),
        CompetitorProfile("Optional competitor", 500, 700, 900, 0.5),
    ]

    checks = []

    normalized = normalize_cost_dataframe(sample)
    validate_cost_dataframe(normalized)
    checks.append(("cost_validation", True, "Triangular cost inputs validated successfully."))

    mean_test = triangular_mean(1, 2, 4)
    checks.append(("triangular_mean", np.isclose(mean_test, 7 / 3), f"Expected 2.3333, got {mean_test:.4f}"))

    std_test = triangular_std(1, 2, 4)
    checks.append(("triangular_std", std_test > 0, f"Expected positive standard deviation, got {std_test:.4f}"))

    draws = simulate_total_project_cost(normalized, simulations=5000, random_seed=42)
    checks.append(("cost_draws_shape", len(draws) == 5000, f"Expected 5000 draws, got {len(draws)}"))
    checks.append(("cost_draws_nonnegative", bool((draws >= 0).all()), "All simulated project costs should be non-negative."))

    bid_results, details = evaluate_bid_grid(
        cost_df=normalized,
        competitors=competitors,
        bid_grid=[650, 700, 750, 800],
        simulations=4000,
        random_seed=123,
        fixed_overhead=50,
    )
    checks.append(("result_rows", len(bid_results) == 4, f"Expected 4 bid rows, got {len(bid_results)}"))
    checks.append(
        (
            "win_probability_bounds",
            bool(bid_results["win_probability"].between(0, 1).all()),
            "All win probabilities should be between 0 and 1.",
        )
    )
    checks.append(
        (
            "details_nonempty",
            not details.empty,
            "Scenario-level detail table should contain simulated outcomes.",
        )
    )

    diagnostics = pd.DataFrame(checks, columns=["check", "passed", "message"])
    return diagnostics
