from __future__ import annotations

from io import StringIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from simulation import (
    CompetitorProfile,
    evaluate_bid_grid,
    normalize_cost_dataframe,
    run_diagnostics,
    summarize_cost_inputs,
    validate_competitors,
    validate_cost_dataframe,
)


st.set_page_config(
    page_title="Miller Construction RFP Analyzer",
    page_icon="🏗️",
    layout="wide",
)


DEFAULT_CSV = """item,low,mode,high,quantity
Site prep,45000,52000,62000,1
Concrete,85000,97000,118000,1
Structural steel,130000,148000,182000,1
Electrical,52000,61000,76000,1
Finishes,38000,45000,57000,1
"""


def currency(value: float) -> str:
    return f"${value:,.0f}"


def percentage(value: float) -> str:
    return f"{value:.1%}"


def _normalize_text(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _find_matching_column(columns: list[str], options: list[str]) -> str | None:
    normalized = {_normalize_text(col): col for col in columns}
    for option in options:
        if option in normalized:
            return normalized[option]
    return None


def _historical_columns(raw_df: pd.DataFrame) -> tuple[str | None, str | None]:
    raw_df = raw_df.copy()
    raw_df.columns = [str(col).strip() for col in raw_df.columns]
    prep_col = _find_matching_column(
        list(raw_df.columns),
        [
            "bid_preparation_costs",
            "bid_prep_costs",
            "bid_preparation_cost",
            "proposal_costs",
            "proposal_cost",
        ],
    )
    total_col = _find_matching_column(
        list(raw_df.columns),
        [
            "total_project_costs_(exluding_bid_preparation_costs)",
            "total_project_costs_(excluding_bid_preparation_costs)",
            "total_project_costs",
            "project_cost",
            "project_costs",
        ],
    )
    return prep_col, total_col


def load_uploaded_data(uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    if uploaded_file is None:
        raw_df = pd.read_csv(StringIO(DEFAULT_CSV))
        return normalize_cost_dataframe(raw_df), None, False

    raw_df = pd.read_csv(uploaded_file)
    try:
        return normalize_cost_dataframe(raw_df), None, False
    except ValueError:
        prep_col, total_col = _historical_columns(raw_df)
        if prep_col is None or total_col is None:
            raise
        return pd.DataFrame(), raw_df, True


def summarize_historical_data(raw_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    prep_col, total_col = _historical_columns(raw_df)
    totals = pd.to_numeric(raw_df[total_col], errors="coerce").dropna()
    prep_costs = pd.to_numeric(raw_df[prep_col], errors="coerce").dropna() if prep_col else pd.Series(dtype=float)
    if totals.empty:
        raise ValueError("The uploaded historical file does not contain any usable total project cost values.")
    return totals, prep_costs


def choose_stat_value(
    series: pd.Series,
    prefix: str,
    label: str,
    default_option: str,
    manual_default: float,
    min_floor: float = 0.0,
) -> float:
    options = {
        "Observed min": float(series.min()),
        "10th percentile": float(series.quantile(0.10)),
        "25th percentile": float(series.quantile(0.25)),
        "Median": float(series.median()),
        "Mean": float(series.mean()),
        "75th percentile": float(series.quantile(0.75)),
        "90th percentile": float(series.quantile(0.90)),
        "95th percentile": float(series.quantile(0.95)),
        "Observed max": float(series.max()),
    }
    selection = st.sidebar.selectbox(
        label,
        options=list(options.keys()) + ["Manual"],
        index=(list(options.keys()) + ["Manual"]).index(default_option),
        key=f"{prefix}_selection",
    )
    if selection == "Manual":
        return st.sidebar.number_input(
            f"Manual {label}",
            min_value=min_floor,
            value=max(min_floor, float(manual_default)),
            step=1000.0,
            key=f"{prefix}_manual",
        )
    return max(min_floor, options[selection])


def build_historical_cost_inputs(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, float, str]:
    totals, prep_costs = summarize_historical_data(raw_df)
    st.sidebar.subheader("Build cost assumptions from CSV")
    st.sidebar.caption("Choose how the triangular cost distribution should be built from the historical file.")

    project_floor = st.sidebar.number_input(
        "Project cost floor ($)",
        min_value=0.0,
        value=70000.0,
        step=1000.0,
        help="The case notes that total project completion costs cannot be less than $70,000.",
    )
    low = choose_stat_value(
        totals,
        prefix="cost_low",
        label="Low cost source",
        default_option="Observed min",
        manual_default=float(totals.min()),
        min_floor=project_floor,
    )
    mode = choose_stat_value(
        totals,
        prefix="cost_mode",
        label="Mode cost source",
        default_option="Median",
        manual_default=float(totals.median()),
        min_floor=project_floor,
    )
    high = choose_stat_value(
        totals,
        prefix="cost_high",
        label="High cost source",
        default_option="Observed max",
        manual_default=float(totals.max()),
        min_floor=project_floor,
    )

    overhead_source = "Zero"
    inferred_overhead = 0.0
    if not prep_costs.empty:
        overhead_source = st.sidebar.selectbox(
            "Bid-preparation overhead source",
            options=["Average from CSV", "Median from CSV", "Observed max", "Manual", "Zero"],
            index=0,
        )
        overhead_map = {
            "Average from CSV": float(prep_costs.mean()),
            "Median from CSV": float(prep_costs.median()),
            "Observed max": float(prep_costs.max()),
            "Zero": 0.0,
        }
        if overhead_source == "Manual":
            inferred_overhead = st.sidebar.number_input(
                "Manual bid-preparation overhead ($)",
                min_value=0.0,
                value=float(prep_costs.mean()),
                step=100.0,
            )
        else:
            inferred_overhead = overhead_map[overhead_source]

    if low > mode or mode > high:
        raise ValueError("Your chosen CSV-based assumptions must satisfy low <= mode <= high.")

    cost_df = pd.DataFrame(
        {
            "item": ["Historical total project cost"],
            "low": [low],
            "mode": [mode],
            "high": [high],
            "quantity": [1.0],
        }
    )
    note = (
        "This cost model was built from the uploaded historical CSV using your selected summary rules. "
        f"Overhead source: {overhead_source}."
    )
    return cost_df, inferred_overhead, note


def build_competitor_profiles(count: int) -> list[CompetitorProfile]:
    competitors: list[CompetitorProfile] = []
    for index in range(count):
        default_probability = 1.0 if index == 0 else 0.5
        st.sidebar.markdown(f"**Competitor {index + 1}**")
        col1, col2, col3 = st.sidebar.columns(3)
        low = col1.number_input(
            f"Low bid #{index + 1}",
            min_value=0.0,
            value=90000.0,
            step=1000.0,
            key=f"comp_low_{index}",
        )
        mode = col2.number_input(
            f"Mode bid #{index + 1}",
            min_value=0.0,
            value=130000.0,
            step=1000.0,
            key=f"comp_mode_{index}",
        )
        high = col3.number_input(
            f"High bid #{index + 1}",
            min_value=0.0,
            value=180000.0,
            step=1000.0,
            key=f"comp_high_{index}",
        )
        participation_pct = st.sidebar.slider(
            f"Chance competitor #{index + 1} bids (%)",
            min_value=0,
            max_value=100,
            value=int(default_probability * 100),
            step=5,
            key=f"comp_prob_{index}",
        )
        competitors.append(
            CompetitorProfile(
                name=f"Competitor {index + 1}",
                low_bid=low,
                mode_bid=mode,
                high_bid=high,
                participation_probability=participation_pct / 100.0,
            )
        )
    return competitors


def historical_summary_table(totals: pd.Series, prep_costs: pd.Series) -> pd.DataFrame:
    rows = [
        ("Observed min", float(totals.min())),
        ("10th percentile", float(totals.quantile(0.10))),
        ("25th percentile", float(totals.quantile(0.25))),
        ("Median", float(totals.median())),
        ("Mean", float(totals.mean())),
        ("75th percentile", float(totals.quantile(0.75))),
        ("90th percentile", float(totals.quantile(0.90))),
        ("Observed max", float(totals.max())),
    ]
    summary = pd.DataFrame(rows, columns=["Statistic", "Total project cost"])
    if not prep_costs.empty:
        summary["Bid prep cost reference"] = [
            float(prep_costs.min()),
            float(prep_costs.quantile(0.10)),
            float(prep_costs.quantile(0.25)),
            float(prep_costs.median()),
            float(prep_costs.mean()),
            float(prep_costs.quantile(0.75)),
            float(prep_costs.quantile(0.90)),
            float(prep_costs.max()),
        ]
    return summary


def main() -> None:
    st.title("Miller Construction RFP Analyzer")
    st.caption(
        "Monte Carlo bidding support for Miller Construction using triangular distributions "
        "for uncertain project costs and competitor bids."
    )

    with st.sidebar:
        st.header("Inputs")
        uploaded_file = st.file_uploader("Upload a project cost CSV", type=["csv"])
        simulations = st.slider("Simulation runs", min_value=1000, max_value=100000, value=50000, step=1000)
        fixed_overhead = st.number_input("Additional fixed overhead ($)", min_value=0.0, value=0.0, step=100.0)
        random_seed = st.number_input("Random seed", min_value=1, value=42, step=1)

    try:
        cost_df, historical_raw_df, historical_mode = load_uploaded_data(uploaded_file)
        inferred_overhead = 0.0
        upload_note = None
        totals = pd.Series(dtype=float)
        prep_costs = pd.Series(dtype=float)
        if historical_mode:
            totals, prep_costs = summarize_historical_data(historical_raw_df)
            cost_df, inferred_overhead, upload_note = build_historical_cost_inputs(historical_raw_df)
        validate_cost_dataframe(cost_df)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.expander("Review or edit the cost table", expanded=uploaded_file is not None):
        edited_cost_df = st.data_editor(
            cost_df,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
        )
    try:
        cost_df = normalize_cost_dataframe(edited_cost_df)
        validate_cost_dataframe(cost_df)
    except Exception as exc:
        st.error(f"Edited cost table is invalid: {exc}")
        st.stop()

    cost_summary = summarize_cost_inputs(cost_df)
    base_expected_cost = float(cost_summary["expected_total_cost"].sum())
    total_overhead = float(fixed_overhead) + float(inferred_overhead)
    expected_cost_with_overhead = base_expected_cost + total_overhead

    with st.sidebar:
        st.subheader("Miller bid search")
        default_bid_min = 90000.0 if historical_mode else round(expected_cost_with_overhead * 0.90, -3)
        default_bid_max = 180000.0 if historical_mode else round(expected_cost_with_overhead * 1.30, -3)
        bid_min = st.number_input("Min bid ($)", min_value=0.0, value=default_bid_min, step=1000.0)
        bid_max = st.number_input("Max bid ($)", min_value=1000.0, value=max(default_bid_max, default_bid_min + 1000.0), step=1000.0)
        bid_step = st.number_input("Step ($)", min_value=100.0, value=1000.0, step=100.0)

        st.subheader("Competitors")
        competitor_count = st.slider("Number of competitors", min_value=1, max_value=8, value=3)

    competitors = build_competitor_profiles(competitor_count)
    try:
        validate_competitors(competitors)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if upload_note:
        st.info(
            f"{upload_note} Inferred bid-preparation overhead from the CSV: {currency(inferred_overhead)}."
        )
    if bid_min >= bid_max:
        st.error("Min bid must be smaller than max bid.")
        st.stop()

    bid_grid = np.arange(bid_min, bid_max + bid_step, bid_step).round(2).tolist()
    results, details = evaluate_bid_grid(
        cost_df=cost_df,
        competitors=competitors,
        bid_grid=bid_grid,
        simulations=simulations,
        random_seed=int(random_seed),
        fixed_overhead=total_overhead,
    )
    best_row = results.iloc[0]

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Expected Project Cost", currency(expected_cost_with_overhead))
    metric2.metric("Recommended Bid", currency(best_row["bid_amount"]))
    metric3.metric("Winning Probability", percentage(best_row["win_probability"]))
    metric4.metric("Expected Profit", currency(best_row["expected_profit"]))

    tab1, tab2, tab3, tab4 = st.tabs(["Recommendation", "Cost Model", "Scenario Table", "Diagnostics"])

    with tab1:
        st.subheader("Best bid recommendation")
        st.write(
            "This recommendation maximizes Miller's unconditional expected profit given the current "
            "cost assumptions, competitor bids, and participation probabilities."
        )

        rec1, rec2, rec3 = st.columns(3)
        rec1.metric("Implied Markup", f"{best_row['implied_markup_pct']:.1f}%")
        rec2.metric("Profit If Won", currency(best_row["expected_profit_if_won"]))
        rec3.metric("Loss Risk If Won", percentage(best_row["loss_probability_if_won"]))

        profit_chart = (
            alt.Chart(results)
            .mark_line(point=True)
            .encode(
                x=alt.X("bid_amount:Q", title="Miller bid ($)"),
                y=alt.Y("expected_profit:Q", title="Expected profit ($)"),
                tooltip=["bid_amount", "implied_markup_pct", "win_probability", "expected_profit"],
            )
            .properties(height=320)
        )
        st.altair_chart(profit_chart, use_container_width=True)

        win_chart = (
            alt.Chart(results)
            .mark_line(color="#0f766e", point=True)
            .encode(
                x=alt.X("bid_amount:Q", title="Miller bid ($)"),
                y=alt.Y("win_probability:Q", title="Win probability"),
                tooltip=["bid_amount", "win_probability"],
            )
            .properties(height=320)
        )
        st.altair_chart(win_chart, use_container_width=True)

        selected_bid = st.select_slider(
            "Inspect a specific bid scenario",
            options=[float(value) for value in results.sort_values("bid_amount")["bid_amount"]],
            value=float(best_row["bid_amount"]),
        )
        selected = results.loc[results["bid_amount"] == selected_bid].iloc[0]
        st.write(
            f"At **{currency(selected_bid)}**, Miller would win about **{percentage(selected['win_probability'])}** "
            f"of the time and earn **{currency(selected['expected_profit'])}** in expected profit."
        )

    with tab2:
        st.subheader("Cost assumptions")
        display_df = cost_summary[
            ["item", "low", "mode", "high", "quantity", "expected_unit_cost", "expected_total_cost"]
        ].copy()
        st.dataframe(display_df, use_container_width=True)

        if historical_mode:
            st.subheader("Historical CSV summary")
            st.dataframe(historical_summary_table(totals, prep_costs), use_container_width=True)
            st.caption(
                "Use the sidebar selectors to choose which historical statistics define the triangular cost model."
            )

        cost_chart = (
            alt.Chart(cost_summary)
            .mark_bar(color="#1d4ed8")
            .encode(
                x=alt.X("expected_total_cost:Q", title="Expected total cost ($)"),
                y=alt.Y("item:N", sort="-x", title="Cost item"),
                tooltip=["item", "expected_total_cost", "quantity"],
            )
            .properties(height=320)
        )
        st.altair_chart(cost_chart, use_container_width=True)

        st.download_button(
            "Download template CSV",
            data=DEFAULT_CSV,
            file_name="miller_project_cost_template.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Optimization table")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download optimization results as CSV",
            data=results.to_csv(index=False),
            file_name="miller_bid_optimization_results.csv",
            mime="text/csv",
        )

        st.subheader("Scenario-level simulation sample")
        selected_detail = details.loc[details["bid_amount"] == selected_bid]
        sample_detail = selected_detail.sample(n=min(500, len(selected_detail)), random_state=1).sort_values(
            "realized_profit"
        )
        scatter = (
            alt.Chart(sample_detail)
            .mark_circle(size=50, opacity=0.5)
            .encode(
                x=alt.X("project_cost:Q", title="Realized project cost ($)"),
                y=alt.Y("realized_profit:Q", title="Realized profit ($)"),
                color=alt.Color("won_project:N", title="Won project"),
                tooltip=["project_cost", "lowest_competitor_bid", "realized_profit", "won_project"],
            )
            .properties(height=350)
        )
        st.altair_chart(scatter, use_container_width=True)

    with tab4:
        st.subheader("Built-in diagnostics")
        diagnostics = run_diagnostics()
        st.dataframe(diagnostics, use_container_width=True)
        if diagnostics["passed"].all():
            st.success("All built-in model checks passed.")
        else:
            st.warning("One or more diagnostic checks failed. Review the table above.")

    st.markdown("---")
    st.write(
        "Tip: with the historical case CSV, you can now decide for yourself which statistics from the data become "
        "the triangular low, mode, and high inputs instead of accepting one automatic choice."
    )


if __name__ == "__main__":
    main()
