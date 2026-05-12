from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import (
    DATA_FILE,
    apply_filters,
    load_raw,
    prep_data,
    safe_group_mean,
    safe_group_sum,
    sidebar_filters,
)


def main() -> None:
    st.set_page_config(page_title="Product", layout="wide")
    st.title("Product analysis")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    left, right = st.columns(2)

    with left:
        st.subheader("PC makes by net sales")
        by_make = safe_group_sum(f, "PC Make", "net_sale").head(20)
        if len(by_make):
            st.bar_chart(by_make.set_index("PC Make")["net_sale"], height=320)
        else:
            st.info("No PC Make / sales data available.")

        st.subheader("Storage type mix")
        if "Storage Type" in f.columns:
            mix = f["Storage Type"].fillna("(missing)").astype(str).value_counts().head(20)
            st.bar_chart(mix, height=320)
        else:
            st.info("No Storage Type column found.")

    with right:
        st.subheader("Top models (net sales)")
        if "PC Model" in f.columns:
            by_model = safe_group_sum(f, "PC Model", "net_sale").head(20)
            if len(by_model):
                st.bar_chart(by_model.set_index("PC Model")["net_sale"], height=320)
                st.dataframe(by_model, use_container_width=True, hide_index=True)
            else:
                st.info("No model/sales data available under current filters.")
        else:
            st.info("No PC Model column found.")

        st.subheader("Average discount by make")
        by_make_disc = safe_group_mean(f, "PC Make", "discount_amount").head(20)
        if len(by_make_disc):
            st.bar_chart(by_make_disc.set_index("PC Make")["discount_amount"], height=320)
        else:
            st.info("No discount data available.")

    st.divider()

    st.subheader("RAM / Storage capacity frequency")
    c1, c2 = st.columns(2)
    with c1:
        if "RAM" in f.columns:
            ram = f["RAM"].fillna("(missing)").astype(str).value_counts().head(20)
            st.bar_chart(ram, height=280)
        else:
            st.info("No RAM column found.")

    with c2:
        if "Storage Capacity" in f.columns:
            cap = f["Storage Capacity"].fillna("(missing)").astype(str).value_counts().head(20)
            st.bar_chart(cap, height=280)
        else:
            st.info("No Storage Capacity column found.")


if __name__ == "__main__":
    main()
