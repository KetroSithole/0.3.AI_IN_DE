from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from data_utils import DATA_FILE


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _parse_date(series: pd.Series) -> pd.Series:
    # Handles values like 'N/A'
    return pd.to_datetime(series.replace({"N/A": pd.NA, "": pd.NA}), errors="coerce").dt.date


@dataclass(frozen=True)
class DimensionResult:
    name: str
    id_col: str
    columns: list[str]
    table: pd.DataFrame
    key_series: pd.Series  # aligned to source df rows


def build_dimension(
    df: pd.DataFrame,
    name: str,
    columns: list[str],
    *,
    start_id: int = 1,
    dropna: bool = False,
) -> DimensionResult | None:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None

    # Create a stable key over the selected columns
    base = df[cols].copy()

    if dropna:
        dim_unique = base.dropna().drop_duplicates()
    else:
        dim_unique = base.drop_duplicates()

    id_col = f"{_slugify(name)}_id"
    dim_unique = dim_unique.reset_index(drop=True)
    dim_unique.insert(0, id_col, range(start_id, start_id + len(dim_unique)))

    # Map each source row to a dimension id
    mapped = base.merge(dim_unique, on=cols, how="left")[id_col]

    return DimensionResult(
        name=name,
        id_col=id_col,
        columns=cols,
        table=dim_unique,
        key_series=mapped,
    )


def build_date_dimension(df: pd.DataFrame, name: str, column: str) -> DimensionResult | None:
    if column not in df.columns:
        return None

    dates = _parse_date(df[column])
    tmp = pd.DataFrame({column: dates})

    dim = tmp.drop_duplicates().sort_values(column).reset_index(drop=True)
    id_col = f"{_slugify(name)}_id"
    dim.insert(0, id_col, range(1, 1 + len(dim)))

    # Add date parts (keep minimal, but useful)
    dim["year"] = pd.to_datetime(dim[column]).dt.year
    dim["month"] = pd.to_datetime(dim[column]).dt.month
    dim["day"] = pd.to_datetime(dim[column]).dt.day
    dim["day_name"] = pd.to_datetime(dim[column]).dt.day_name()

    mapped = tmp.merge(dim[[id_col, column]], on=column, how="left")[id_col]

    return DimensionResult(
        name=name,
        id_col=id_col,
        columns=[column],
        table=dim,
        key_series=mapped,
    )


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_resource(show_spinner=False)
def build_star_schema(df: pd.DataFrame) -> tuple[dict[str, DimensionResult], pd.DataFrame]:
    dims: dict[str, DimensionResult] = {}

    # Predefined dimension sets based on the columns we see in the file header
    candidates: list[tuple[str, list[str], dict[str, Any]]] = [
        ("Geography", ["Continent", "Country or State", "Province or City"], {}),
        ("Shop", ["Shop Name", "Shop Age"], {}),
        ("PC", ["PC Make", "PC Model", "Storage Type", "Storage Capacity", "RAM"], {}),
        (
            "Customer",
            [
                "Customer Name",
                "Customer Surname",
                "Customer Contact Number",
                "Customer Email Address",
            ],
            {},
        ),
        ("Salesperson", ["Sales Person Name", "Sales Person Department"], {}),
        ("Payment", ["Payment Method"], {}),
        ("Channel", ["Channel"], {}),
        ("Priority", ["Priority"], {}),
    ]

    for dim_name, cols, kwargs in candidates:
        res = build_dimension(df, dim_name, cols, **kwargs)
        if res is not None:
            dims[dim_name] = res

    # Date dimensions
    for dim_name, col in [("Purchase Date", "Purchase Date"), ("Ship Date", "Ship Date")]:
        res = build_date_dimension(df, dim_name, col)
        if res is not None:
            dims[dim_name] = res

    # Build the fact table: keep the raw measures + add foreign keys
    fact = pd.DataFrame({"transaction_id": range(1, 1 + len(df))})

    # Foreign keys
    for dim in dims.values():
        fact[dim.id_col] = dim.key_series

    # Measures (only include columns that exist)
    measure_cols = [
        "Cost Price",
        "Sale Price",
        "Discount Amount",
        "Finance Amount",
        "Cost of Repairs",
        "Total Sales per Employee",
        "PC Market Price",
        "Credit Score",
    ]
    for col in measure_cols:
        if col in df.columns:
            fact[col] = df[col]

    return dims, fact


def main() -> None:
    st.set_page_config(page_title="PC Sales Dashboard", layout="wide")

    st.title("PC Sales Dashboard")

    if not DATA_FILE.exists():
        st.error(f"Could not find data file: {DATA_FILE}")
        st.stop()

    df = load_data(DATA_FILE)
    dims, fact = build_star_schema(df)

    st.sidebar.header("Dataset")
    st.sidebar.write("File:", str(DATA_FILE.name))
    st.sidebar.metric("Rows", f"{len(df):,}")
    st.sidebar.metric("Columns", f"{df.shape[1]:,}")

    tab_dims, tab_fact = st.tabs(["Dimensions", "Fact Table"])

    with tab_dims:
        st.subheader("Dimension tables")
        st.caption(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        if not dims:
            st.info("No dimension tables could be generated from the current CSV.")
        else:
            dim_names = list(dims.keys())
            selected = st.selectbox("Select a dimension", dim_names)
            dim = dims[selected]

            dim_table = dim.table.copy()
            if selected == "Customer":
                if "Customer Contact Number" in dim_table.columns:
                    s = dim_table["Customer Contact Number"].astype(str)
                    last4 = s.str.replace(r"\D", "", regex=True).str[-4:]
                    dim_table["Customer Contact Number"] = last4.apply(
                        lambda x: f"***{x}" if isinstance(x, str) and x else pd.NA
                    )
                if "Customer Email Address" in dim_table.columns:
                    email = dim_table["Customer Email Address"].astype(str)
                    parts = email.str.split("@", n=1, expand=True)
                    if parts.shape[1] == 2:
                        dim_table["Customer Email Address"] = "***@" + parts[1].str.lower()
                if "Customer Surname" in dim_table.columns:
                    dim_table["Customer Surname"] = dim_table["Customer Surname"].astype(str).str[:1] + "."

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dimension rows", f"{len(dim.table):,}")
            with col2:
                st.write("Columns:", ", ".join(dim.columns))

            st.dataframe(dim_table, use_container_width=True)

            st.download_button(
                label=f"Download {selected} dimension (CSV)",
                data=dim_table.to_csv(index=False).encode("utf-8"),
                file_name=f"dimension_{_slugify(selected)}.csv",
                mime="text/csv",
            )

    with tab_fact:
        st.subheader("Fact table")
        st.write(
            "This fact table keeps the measures and adds foreign keys to the dimensions above."
        )
        st.dataframe(fact.head(200), use_container_width=True)
        st.caption(f"Fact table shape: {fact.shape[0]:,} rows × {fact.shape[1]:,} columns")

        st.download_button(
            label="Download fact table (CSV)",
            data=fact.to_csv(index=False).encode("utf-8"),
            file_name="fact_table.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
