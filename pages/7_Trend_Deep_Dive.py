from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, prep_data, sidebar_filters


def _time_bucket(df: pd.DataFrame, granularity: str) -> pd.Series:
    d = df["purchase_date"]
    if granularity == "Daily":
        return d.dt.floor("D")
    if granularity == "Weekly":
        # Week starting Monday
        return d.dt.to_period("W-MON").dt.start_time
    # Monthly
    return d.dt.to_period("M").dt.to_timestamp()


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def main() -> None:
    st.set_page_config(page_title="Trend Deep Dive", layout="wide")
    st.title("Trend deep dive")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state).copy()

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    if len(f) == 0:
        st.info("No rows match the current filters.")
        return

    if "purchase_date" not in f.columns or f["purchase_date"].dropna().empty:
        st.info("No valid Purchase Date values available for trend analysis.")
        return

    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        granularity = st.radio("Time grain", ["Monthly", "Weekly", "Daily"], horizontal=True)
    with c2:
        breakdown = st.selectbox(
            "Breakdown",
            ["Overall", "Channel", "Continent", "PC Make", "Payment Method"],
        )
    with c3:
        top_n = st.slider("Top groups", min_value=3, max_value=20, value=8)

    f["t"] = _time_bucket(f, granularity)
    f = f.dropna(subset=["t"])

    # Measure selection
    st.subheader("Measure")
    measure = st.selectbox(
        "Trend measure",
        [
            ("Net sales", "net_sale"),
            ("Profit after repairs", "profit_after_repairs"),
            ("Gross profit", "gross_profit"),
            ("Discount amount", "discount_amount"),
        ],
        format_func=lambda x: x[0],
    )
    measure_col = measure[1]

    f[measure_col] = _safe_numeric(f, measure_col)

    st.divider()

    if breakdown == "Overall":
        agg = (
            f.groupby("t", as_index=False)[measure_col]
            .sum(min_count=1)
            .sort_values("t")
        )
        if agg.empty:
            st.info("No values available for the selected measure.")
            return

        agg["rolling_3"] = agg[measure_col].rolling(3, min_periods=1).mean()
        agg["growth_%"] = agg[measure_col].pct_change() * 100

        left, right = st.columns(2)

        with left:
            st.subheader("Trend (with 3-period rolling average)")
            st.line_chart(agg.set_index("t")[[measure_col, "rolling_3"]], height=340)

        with right:
            st.subheader("Period-over-period growth (%)")
            growth = agg.set_index("t")["growth_%"].dropna()
            if len(growth):
                st.line_chart(growth, height=340)
            else:
                st.info("Not enough periods to compute growth.")

        st.subheader("Best / worst periods")
        best = agg.dropna(subset=[measure_col]).nlargest(5, measure_col)
        worst = agg.dropna(subset=[measure_col]).nsmallest(5, measure_col)
        b1, b2 = st.columns(2)
        with b1:
            st.write("Top 5")
            st.dataframe(best, use_container_width=True, hide_index=True)
        with b2:
            st.write("Bottom 5")
            st.dataframe(worst, use_container_width=True, hide_index=True)

        st.subheader("All periods")
        st.dataframe(agg, use_container_width=True, hide_index=True)

    else:
        group_col = breakdown
        if group_col not in f.columns:
            st.info(f"Column '{group_col}' not found in this dataset.")
            return

        tmp = f.dropna(subset=[group_col])
        if tmp.empty:
            st.info("No rows available for the selected breakdown.")
            return

        # Keep only top groups by total measure
        totals = (
            tmp.groupby(group_col, as_index=False)[measure_col]
            .sum(min_count=1)
            .sort_values(measure_col, ascending=False)
        )
        keep = totals[group_col].head(top_n).tolist()
        tmp = tmp[tmp[group_col].isin(keep)]

        agg = (
            tmp.groupby(["t", group_col], as_index=False)[measure_col]
            .sum(min_count=1)
            .sort_values("t")
        )

        pivot = agg.pivot(index="t", columns=group_col, values=measure_col).fillna(0)
        st.subheader(f"{measure[0]} over time by {breakdown} (top {len(keep)})")
        st.line_chart(pivot, height=380)

        st.subheader("Totals by group")
        st.dataframe(totals.head(top_n), use_container_width=True, hide_index=True)

        st.subheader("Detailed table")
        st.dataframe(agg, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
