from __future__ import annotations

import pandas as pd
import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, prep_data, sidebar_filters


def main() -> None:
    st.set_page_config(page_title="Data Quality", layout="wide")
    st.title("Data quality")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    st.subheader("Missing values by column")
    miss = pd.DataFrame(
        {
            "column": f.columns,
            "missing": f.isna().sum().values,
            "missing_%": (f.isna().mean().values * 100).round(2),
            "unique": [f[c].nunique(dropna=True) for c in f.columns],
        }
    ).sort_values(["missing_%", "missing"], ascending=False)
    st.dataframe(miss, use_container_width=True, hide_index=True)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Date health")
        if "purchase_date" in f.columns:
            invalid_purchase = int(f["purchase_date"].isna().sum())
            st.metric("Missing/invalid Purchase Date", f"{invalid_purchase:,}")
        if "ship_date" in f.columns:
            invalid_ship = int(f["ship_date"].isna().sum())
            st.metric("Missing/invalid Ship Date", f"{invalid_ship:,}")

        if "ship_days" in f.columns:
            ship_days = pd.to_numeric(f["ship_days"], errors="coerce")
            st.metric("Negative ship days", f"{int((ship_days < 0).sum()):,}")

    with right:
        st.subheader("Duplicates")
        # A loose duplicate check using a set of business-ish columns (if present)
        key_cols = [
            c
            for c in [
                "Purchase Date",
                "Customer Email Address",
                "PC Make",
                "PC Model",
                "Shop Name",
                "Sale Price",
            ]
            if c in raw.columns
        ]
        if key_cols:
            dupes = int(raw.duplicated(subset=key_cols, keep=False).sum())
            st.metric("Potential duplicate rows", f"{dupes:,}")
            st.caption("Duplicate check uses: " + ", ".join(key_cols))
        else:
            st.info("Not enough columns available to run duplicate checks.")

    st.divider()

    st.subheader("Outlier checks (quick)")
    checks = []
    for col in ["sale_price", "cost_price", "discount_amount", "finance_amount", "credit_score", "cost_of_repairs"]:
        if col in f.columns:
            s = pd.to_numeric(f[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            q1 = float(s.quantile(0.25))
            q3 = float(s.quantile(0.75))
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
            outliers = int(((s < lo) | (s > hi)).sum())
            checks.append(
                {
                    "field": col,
                    "rows": int(len(s)),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "outliers_(IQR)": outliers,
                }
            )

    if checks:
        st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)
    else:
        st.info("No numeric fields available for outlier checks.")


if __name__ == "__main__":
    main()
