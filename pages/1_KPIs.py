from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import (
    DATA_FILE,
    apply_filters,
    kpi,
    load_raw,
    prep_data,
    safe_group_sum,
    sidebar_filters,
)


def main() -> None:
    st.set_page_config(page_title="KPIs", layout="wide")
    st.title("KPIs")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    c1, c2, c3, c4 = st.columns(4)

    total_net_sale = float(pd.to_numeric(f.get("net_sale"), errors="coerce").sum())
    total_profit = float(pd.to_numeric(f.get("profit_after_repairs"), errors="coerce").sum())
    total_repairs = float(pd.to_numeric(f.get("cost_of_repairs"), errors="coerce").sum())
    avg_discount = float(pd.to_numeric(f.get("discount_amount"), errors="coerce").mean())

    c1.metric("Transactions", f"{len(f):,}")
    c2.metric("Net sales", kpi(total_net_sale))
    c3.metric("Profit after repairs", kpi(total_profit))
    c4.metric("Avg discount", kpi(avg_discount))

    c5, c6, c7, c8 = st.columns(4)
    avg_credit = float(pd.to_numeric(f.get("credit_score"), errors="coerce").mean())
    finance_sum = float(pd.to_numeric(f.get("finance_amount"), errors="coerce").sum())

    c5.metric("Repair costs", kpi(total_repairs))
    c6.metric("Finance amount", kpi(finance_sum))
    c7.metric("Avg credit score", kpi(avg_credit))
    c8.metric(
        "Avg ship days",
        kpi(pd.to_numeric(f.get("ship_days"), errors="coerce").mean()),
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Net sales by channel")
        by_channel = safe_group_sum(f, "Channel", "net_sale")
        if len(by_channel):
            st.bar_chart(by_channel.set_index("Channel")["net_sale"], height=320)
        else:
            st.info("No channel data available for the current filters.")

        st.subheader("Net sales by payment method")
        by_pay = safe_group_sum(f, "Payment Method", "net_sale")
        if len(by_pay):
            st.bar_chart(by_pay.set_index("Payment Method")["net_sale"], height=320)
        else:
            st.info("No payment method data available for the current filters.")

    with right:
        st.subheader("Top PC makes (net sales)")
        by_make = safe_group_sum(f, "PC Make", "net_sale").head(15)
        if len(by_make):
            st.bar_chart(by_make.set_index("PC Make")["net_sale"], height=320)
            st.dataframe(by_make, use_container_width=True, hide_index=True)
        else:
            st.info("No PC make data available for the current filters.")

        st.subheader("Top shops (net sales)")
        by_shop = safe_group_sum(f, "Shop Name", "net_sale").head(15)
        if len(by_shop):
            st.bar_chart(by_shop.set_index("Shop Name")["net_sale"], height=320)
            st.dataframe(by_shop, use_container_width=True, hide_index=True)
        else:
            st.info("No shop data available for the current filters.")


if __name__ == "__main__":
    main()
