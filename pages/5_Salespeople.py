from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, prep_data, sidebar_filters


def _agg_salesperson(f: pd.DataFrame) -> pd.DataFrame:
    if "Sales Person Name" not in f.columns:
        return pd.DataFrame()

    cols = [c for c in ["Sales Person Name", "Sales Person Department", "net_sale", "gross_profit", "profit_after_repairs", "discount_amount"] if c in f.columns]
    tmp = f[cols].copy()

    num_cols = [c for c in ["net_sale", "gross_profit", "profit_after_repairs", "discount_amount"] if c in tmp.columns]
    for c in num_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    grouped = tmp.groupby([c for c in ["Sales Person Name", "Sales Person Department"] if c in tmp.columns], as_index=False).agg(
        transactions=("Sales Person Name", "count"),
        net_sales=("net_sale", "sum"),
        profit_after_repairs=("profit_after_repairs", "sum"),
        avg_discount=("discount_amount", "mean"),
    )

    return grouped.sort_values("net_sales", ascending=False)


def main() -> None:
    st.set_page_config(page_title="Salespeople", layout="wide")
    st.title("Salespeople")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    table = _agg_salesperson(f)
    if table.empty:
        st.info("No salesperson fields found in the data.")
        return

    st.subheader("Leaderboard (net sales)")
    st.dataframe(table, use_container_width=True, hide_index=True)

    top_n = st.slider("Top N for chart", min_value=5, max_value=30, value=15)
    top = table.head(top_n)

    st.subheader("Top salespeople by net sales")
    st.bar_chart(top.set_index("Sales Person Name")["net_sales"], height=360)

    if "Sales Person Department" in top.columns:
        st.subheader("Department view")
        dept = top.groupby("Sales Person Department", as_index=True)["net_sales"].sum().sort_values(ascending=False)
        st.bar_chart(dept, height=280)


if __name__ == "__main__":
    main()
