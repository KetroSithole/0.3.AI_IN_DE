from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, month_bucket, prep_data, sidebar_filters


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df.get(col), errors="coerce")


def _credit_band(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    # Common score buckets (kept simple)
    bins = [0, 580, 670, 740, 800, 1000]
    labels = ["0-579 (Poor)", "580-669 (Fair)", "670-739 (Good)", "740-799 (Very good)", "800+ (Excellent)"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=False)


def main() -> None:
    st.set_page_config(page_title="Segmentation", layout="wide")
    st.title("Segmentation")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state).copy()

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    if len(f) == 0:
        st.info("No rows match the current filters.")
        return

    tab_credit, tab_finance, tab_discount, tab_heat = st.tabs(
        ["Credit", "Finance behavior", "Discount & margin", "Month × Channel"]
    )

    with tab_credit:
        st.subheader("Credit-score segmentation")
        if "credit_score" not in f.columns:
            st.info("Credit score is not available in this dataset.")
        else:
            f["credit_band"] = _credit_band(f["credit_score"]).astype(str)
            # Replace NaN band
            f.loc[pd.isna(_num(f, "credit_score")), "credit_band"] = "(missing)"

            agg = (
                f.groupby("credit_band", as_index=False)
                .agg(
                    transactions=("credit_band", "count"),
                    net_sales=("net_sale", "sum"),
                    avg_discount_rate=("discount_rate", "mean"),
                    avg_margin_rate=("margin_rate", "mean"),
                )
                .sort_values("transactions", ascending=False)
            )

            c1, c2 = st.columns(2)
            with c1:
                st.altair_chart(
                    alt.Chart(agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("credit_band:N", sort=None, title="Credit band"),
                        y=alt.Y("transactions:Q", title="Transactions"),
                        tooltip=[
                            alt.Tooltip("credit_band:N"),
                            alt.Tooltip("transactions:Q", format=","),
                            alt.Tooltip("net_sales:Q", format=",.0f"),
                        ],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )
            with c2:
                st.altair_chart(
                    alt.Chart(agg)
                    .mark_bar()
                    .encode(
                        x=alt.X("credit_band:N", sort=None, title="Credit band"),
                        y=alt.Y("net_sales:Q", title="Net sales"),
                        tooltip=[
                            alt.Tooltip("credit_band:N"),
                            alt.Tooltip("net_sales:Q", format=",.0f"),
                            alt.Tooltip("avg_margin_rate:Q", format=".2%"),
                        ],
                    )
                    .properties(height=320),
                    use_container_width=True,
                )

            st.dataframe(agg, use_container_width=True, hide_index=True)

    with tab_finance:
        st.subheader("Finance vs non-finance")
        if "is_financed" not in f.columns:
            st.info("Payment Method not available.")
        else:
            rate = float(pd.Series(f["is_financed"]).mean())
            st.metric("Finance share", f"{rate * 100:,.1f}%")

            left, right = st.columns(2)
            with left:
                if "Channel" in f.columns:
                    tmp = f.groupby("Channel", as_index=False)["is_financed"].mean()
                    tmp["finance_share"] = tmp["is_financed"]
                    chart = (
                        alt.Chart(tmp)
                        .mark_bar()
                        .encode(
                            x=alt.X("Channel:N", title="Channel"),
                            y=alt.Y("finance_share:Q", title="Finance share", axis=alt.Axis(format="%")),
                            tooltip=[alt.Tooltip("Channel:N"), alt.Tooltip("finance_share:Q", format=".1%")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No Channel column available.")

            with right:
                if "credit_score" in f.columns:
                    f["credit_band"] = _credit_band(f["credit_score"]).astype(str)
                    f.loc[pd.isna(_num(f, "credit_score")), "credit_band"] = "(missing)"
                    tmp = f.groupby("credit_band", as_index=False)["is_financed"].mean()
                    tmp["finance_share"] = tmp["is_financed"]
                    chart = (
                        alt.Chart(tmp)
                        .mark_bar()
                        .encode(
                            x=alt.X("credit_band:N", sort=None, title="Credit band"),
                            y=alt.Y("finance_share:Q", title="Finance share", axis=alt.Axis(format="%")),
                            tooltip=[alt.Tooltip("credit_band:N"), alt.Tooltip("finance_share:Q", format=".1%")],
                        )
                        .properties(height=320)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info("No Credit Score column available.")

    with tab_discount:
        st.subheader("Discount rate & margin rate")

        # Build a tidy frame
        cols = [c for c in ["discount_rate", "margin_rate", "net_sale", "Channel", "PC Make", "Payment Method"] if c in f.columns]
        tmp = f[cols].copy() if cols else pd.DataFrame()

        if tmp.empty or ("discount_rate" not in tmp.columns and "margin_rate" not in tmp.columns):
            st.info("Discount/margin fields not available.")
        else:
            seg = st.selectbox("Segment by", [c for c in ["Channel", "PC Make", "Payment Method"] if c in tmp.columns] or ["(none)"])

            if seg == "(none)":
                st.info("No segment columns available for comparison.")
            else:
                agg = (
                    tmp.groupby(seg, as_index=False)
                    .agg(
                        transactions=(seg, "count"),
                        avg_discount_rate=("discount_rate", "mean"),
                        avg_margin_rate=("margin_rate", "mean"),
                        net_sales=("net_sale", "sum"),
                    )
                    .sort_values("net_sales", ascending=False)
                    .head(25)
                )

                c1, c2 = st.columns(2)
                with c1:
                    st.altair_chart(
                        alt.Chart(agg)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{seg}:N", sort=None, title=seg),
                            y=alt.Y("avg_discount_rate:Q", axis=alt.Axis(format="%"), title="Avg discount rate"),
                            tooltip=[
                                alt.Tooltip(f"{seg}:N"),
                                alt.Tooltip("avg_discount_rate:Q", format=".2%"),
                                alt.Tooltip("transactions:Q", format=","),
                            ],
                        )
                        .properties(height=340),
                        use_container_width=True,
                    )
                with c2:
                    st.altair_chart(
                        alt.Chart(agg)
                        .mark_bar()
                        .encode(
                            x=alt.X(f"{seg}:N", sort=None, title=seg),
                            y=alt.Y("avg_margin_rate:Q", axis=alt.Axis(format="%"), title="Avg margin rate"),
                            tooltip=[
                                alt.Tooltip(f"{seg}:N"),
                                alt.Tooltip("avg_margin_rate:Q", format=".2%"),
                                alt.Tooltip("net_sales:Q", format=",.0f"),
                            ],
                        )
                        .properties(height=340),
                        use_container_width=True,
                    )

                st.dataframe(agg, use_container_width=True, hide_index=True)

    with tab_heat:
        st.subheader("Heatmap: month × channel")

        if "purchase_date" not in f.columns or f["purchase_date"].dropna().empty:
            st.info("No valid purchase dates.")
            return
        if "Channel" not in f.columns:
            st.info("No Channel column in dataset.")
            return

        f["month"] = month_bucket(f, "purchase_date")
        tmp = f.dropna(subset=["month"]).copy()
        tmp["net_sale"] = pd.to_numeric(tmp.get("net_sale"), errors="coerce")
        tmp = tmp.dropna(subset=["net_sale"])

        if tmp.empty:
            st.info("No net sales values for this heatmap.")
            return

        agg = tmp.groupby(["month", "Channel"], as_index=False)["net_sale"].sum().sort_values("month")

        heat = (
            alt.Chart(agg)
            .mark_rect()
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("Channel:N", title="Channel"),
                color=alt.Color("net_sale:Q", title="Net sales", scale=alt.Scale(scheme="blues")),
                tooltip=[alt.Tooltip("month:T"), alt.Tooltip("Channel:N"), alt.Tooltip("net_sale:Q", format=",.0f")],
            )
            .properties(height=360)
        )
        st.altair_chart(heat, use_container_width=True)

        st.dataframe(agg, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
