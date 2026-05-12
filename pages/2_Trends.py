from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, month_bucket, prep_data, sidebar_filters


def main() -> None:
    st.set_page_config(page_title="Trends", layout="wide")
    st.title("Trends")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    if len(f) == 0:
        st.info("No rows match the current filters.")
        return

    st.subheader("Monthly net sales")
    f = f.copy()
    f["purchase_month"] = month_bucket(f, "purchase_date")

    monthly = (
        f.dropna(subset=["purchase_month"])
        .groupby("purchase_month", as_index=False)[["net_sale", "gross_profit", "profit_after_repairs"]]
        .sum(numeric_only=True)
        .sort_values("purchase_month")
    )

    if len(monthly):
        chart = monthly.set_index("purchase_month")[
            [c for c in ["net_sale", "gross_profit", "profit_after_repairs"] if c in monthly.columns]
        ]
        st.line_chart(chart, height=320)
        st.dataframe(monthly, use_container_width=True, hide_index=True)
    else:
        st.info("No valid purchase dates to build a monthly trend.")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Shipping lead time")
        ship_days = pd.to_numeric(f.get("ship_days"), errors="coerce").dropna()
        if len(ship_days):
            st.write(
                {
                    "avg_days": float(ship_days.mean()),
                    "p50_days": float(ship_days.median()),
                    "p90_days": float(ship_days.quantile(0.9)),
                }
            )
            st.bar_chart(ship_days.value_counts().sort_index(), height=320)
        else:
            st.info("No shipping date information available (or Ship Date is N/A).")

    with right:
        st.subheader("Discounts over time")
        if "purchase_month" in f.columns and "discount_amount" in f.columns:
            disc = (
                f.dropna(subset=["purchase_month"])
                .groupby("purchase_month", as_index=False)["discount_amount"]
                .mean(numeric_only=True)
                .sort_values("purchase_month")
            )
            if len(disc):
                st.line_chart(disc.set_index("purchase_month")["discount_amount"], height=320)
                st.dataframe(disc, use_container_width=True, hide_index=True)
            else:
                st.info("No discount values available for the current filters.")
        else:
            st.info("Discount data not available in this dataset.")


if __name__ == "__main__":
    main()
