from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, prep_data, sidebar_filters


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _exists(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


def _pareto(df: pd.DataFrame, group_col: str, value_col: str, top_n: int = 15) -> pd.DataFrame:
    tmp = df[[group_col, value_col]].copy()
    tmp[group_col] = tmp[group_col].fillna("(missing)").astype(str)
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])

    agg = tmp.groupby(group_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    agg = agg.head(top_n).reset_index(drop=True)
    total = float(agg[value_col].sum())
    agg["cum_sum"] = agg[value_col].cumsum()
    agg["cum_%"] = np.where(total > 0, agg["cum_sum"] / total * 100, np.nan)
    return agg


def main() -> None:
    st.set_page_config(page_title="Visual Insights", layout="wide")
    st.title("Visual insights")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state).copy()

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    if len(f) == 0:
        st.info("No rows match the current filters.")
        return

    tab_dist, tab_rel, tab_corr, tab_pareto = st.tabs(
        ["Distributions", "Relationships", "Correlations", "Pareto"]
    )

    with tab_dist:
        st.subheader("Distributions")

        candidates = [
            ("Net sale", "net_sale"),
            ("Sale price", "sale_price"),
            ("Cost price", "cost_price"),
            ("Discount amount", "discount_amount"),
            ("Profit after repairs", "profit_after_repairs"),
            ("Credit score", "credit_score"),
            ("Ship days", "ship_days"),
        ]
        available = [(label, col) for (label, col) in candidates if _exists(f, col)]

        if not available:
            st.info("No numeric fields available for distribution plots.")
        else:
            metric = st.selectbox("Pick a metric", available, format_func=lambda x: x[0])
            col = metric[1]
            s = _num(f, col).dropna()

            c1, c2, c3 = st.columns(3)
            c1.metric("Count", f"{len(s):,}")
            if len(s):
                c2.metric("Median", f"{float(s.median()):,.2f}")
                c3.metric("Mean", f"{float(s.mean()):,.2f}")

            bins = st.slider("Histogram bins", min_value=10, max_value=80, value=30)

            hist_df = pd.DataFrame({"value": s})
            chart = (
                alt.Chart(hist_df)
                .mark_bar()
                .encode(
                    x=alt.X("value:Q", bin=alt.Bin(maxbins=bins), title=metric[0]),
                    y=alt.Y("count():Q", title="Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Quick percentiles")
            if len(s):
                p = s.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_frame("value")
                p.index = ["p10", "p25", "p50", "p75", "p90"]
                st.dataframe(p.reset_index(names="percentile"), use_container_width=True, hide_index=True)

    with tab_rel:
        st.subheader("Relationships")

        x_choices = [
            ("Sale price", "sale_price"),
            ("Cost price", "cost_price"),
            ("Discount amount", "discount_amount"),
            ("Finance amount", "finance_amount"),
            ("Credit score", "credit_score"),
            ("Ship days", "ship_days"),
        ]
        y_choices = [
            ("Net sale", "net_sale"),
            ("Profit after repairs", "profit_after_repairs"),
            ("Gross profit", "gross_profit"),
        ]

        x_avail = [(l, c) for (l, c) in x_choices if _exists(f, c)]
        y_avail = [(l, c) for (l, c) in y_choices if _exists(f, c)]

        if not x_avail or not y_avail:
            st.info("Not enough numeric fields for relationship plots.")
        else:
            cx, cy, cc = st.columns(3)
            x = cx.selectbox("X", x_avail, format_func=lambda x: x[0])[1]
            y = cy.selectbox("Y", y_avail, format_func=lambda x: x[0])[1]
            color = cc.selectbox("Color", ["None", "Channel", "Continent", "Payment Method", "PC Make"], index=1)

            plot = f.copy()
            plot[x] = _num(plot, x)
            plot[y] = _num(plot, y)
            plot = plot.dropna(subset=[x, y])

            if len(plot) == 0:
                st.info("No rows available after removing missing values.")
            else:
                base = alt.Chart(plot).mark_circle(size=60, opacity=0.55).encode(
                    x=alt.X(f"{x}:Q"),
                    y=alt.Y(f"{y}:Q"),
                    tooltip=[
                        alt.Tooltip(f"{x}:Q"),
                        alt.Tooltip(f"{y}:Q"),
                    ],
                )

                if color != "None" and color in plot.columns:
                    base = base.encode(color=alt.Color(f"{color}:N"))

                trend = alt.Chart(plot).transform_regression(x, y).mark_line(color="black").encode(
                    x=alt.X(f"{x}:Q"),
                    y=alt.Y(f"{y}:Q"),
                )

                st.altair_chart((base + trend).properties(height=380), use_container_width=True)

                st.subheader("Correlation (Pearson)")
                corr = float(plot[[x, y]].corr(method="pearson").iloc[0, 1])
                st.metric("r", f"{corr:,.3f}")

    with tab_corr:
        st.subheader("Correlations")

        numeric_cols = [
            c
            for c in [
                "sale_price",
                "cost_price",
                "discount_amount",
                "finance_amount",
                "net_sale",
                "gross_profit",
                "profit_after_repairs",
                "credit_score",
                "ship_days",
                "pc_market_price",
                "total_sales_per_employee",
                "cost_of_repairs",
            ]
            if c in f.columns
        ]

        if len(numeric_cols) < 2:
            st.info("Not enough numeric columns to compute correlations.")
        else:
            corr_df = f[numeric_cols].apply(pd.to_numeric, errors="coerce")
            corr = corr_df.corr().fillna(0)

            melted = corr.reset_index(names="x").melt(id_vars=["x"], var_name="y", value_name="corr")

            heat = (
                alt.Chart(melted)
                .mark_rect()
                .encode(
                    x=alt.X("x:N", title=""),
                    y=alt.Y("y:N", title=""),
                    color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue", domain=[-1, 1])),
                    tooltip=[alt.Tooltip("x:N"), alt.Tooltip("y:N"), alt.Tooltip("corr:Q", format=".3f")],
                )
                .properties(height=420)
            )
            st.altair_chart(heat, use_container_width=True)

            st.subheader("Correlation table")
            st.dataframe(corr.round(3), use_container_width=True)

    with tab_pareto:
        st.subheader("Pareto / Top contributors")
        group = st.selectbox(
            "Group by",
            ["PC Make", "PC Model", "Shop Name", "Sales Person Name", "Country or State", "Channel"],
            index=0,
        )
        value = st.selectbox(
            "Value",
            [("Net sales", "net_sale"), ("Profit after repairs", "profit_after_repairs"), ("Discount amount", "discount_amount")],
            format_func=lambda x: x[0],
        )[1]
        top_n = st.slider("Top N", min_value=5, max_value=30, value=15)

        if group not in f.columns or value not in f.columns:
            st.info("Selected group/value not available in the current filtered dataset.")
        else:
            pareto = _pareto(f, group, value, top_n=top_n)
            if pareto.empty:
                st.info("No data available for the selected Pareto view.")
            else:
                bars = alt.Chart(pareto).mark_bar().encode(
                    x=alt.X(f"{group}:N", sort=None, title=group),
                    y=alt.Y(f"{value}:Q", title=value),
                    tooltip=[
                        alt.Tooltip(f"{group}:N"),
                        alt.Tooltip(f"{value}:Q", format=",.0f"),
                        alt.Tooltip("cum_%:Q", format=".1f", title="Cum %"),
                    ],
                )

                line = alt.Chart(pareto).mark_line(color="black").encode(
                    x=alt.X(f"{group}:N", sort=None),
                    y=alt.Y("cum_%:Q", title="Cumulative %"),
                )

                st.altair_chart(
                    alt.layer(bars, line).resolve_scale(y="independent").properties(height=420),
                    use_container_width=True,
                )
                st.dataframe(pareto, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
