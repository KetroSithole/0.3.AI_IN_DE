from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


DATA_FILE = Path(__file__).with_name("pc_data.csv")


def _to_numeric(series: pd.Series) -> pd.Series:
    # Handles commas, blanks, N/A and mixed types.
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.replace({"N/A": pd.NA, "": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_date(series: pd.Series) -> pd.Series:
    cleaned = series.replace({"N/A": pd.NA, "": pd.NA})
    return pd.to_datetime(cleaned, errors="coerce")


@st.cache_data(show_spinner=False)
def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def prep_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Dates
    if "Purchase Date" in out.columns:
        out["purchase_date"] = _parse_date(out["Purchase Date"])
    else:
        out["purchase_date"] = pd.NaT

    if "Ship Date" in out.columns:
        out["ship_date"] = _parse_date(out["Ship Date"])
    else:
        out["ship_date"] = pd.NaT

    # Numerics
    for col in [
        "Cost Price",
        "Sale Price",
        "Discount Amount",
        "Finance Amount",
        "Cost of Repairs",
        "Total Sales per Employee",
        "PC Market Price",
        "Credit Score",
        "Shop Age",
    ]:
        if col in out.columns:
            out[_snake(col)] = _to_numeric(out[col])

    # Derived measures
    cost = out.get("cost_price")
    sale = out.get("sale_price")
    discount = out.get("discount_amount")
    repairs = out.get("cost_of_repairs")

    if sale is not None and discount is not None:
        out["net_sale"] = sale - discount.fillna(0)
    elif sale is not None:
        out["net_sale"] = sale
    else:
        out["net_sale"] = pd.NA

    if cost is not None and out["net_sale"].isna().all() is False:
        out["gross_profit"] = out["net_sale"] - cost
    else:
        out["gross_profit"] = pd.NA

    if repairs is not None and out["gross_profit"].isna().all() is False:
        out["profit_after_repairs"] = out["gross_profit"] - repairs.fillna(0)
    else:
        out["profit_after_repairs"] = pd.NA

    # Shipping lead time
    out["ship_days"] = (out["ship_date"] - out["purchase_date"]).dt.days

    # Light PII-safe helper fields
    if "Customer Email Address" in out.columns:
        email = out["Customer Email Address"].astype(str)
        out["email_domain"] = email.str.split("@").str[-1].str.lower()
        out.loc[email.isna() | (email == "nan"), "email_domain"] = pd.NA

    return out


def _snake(name: str) -> str:
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


@dataclass(frozen=True)
class FilterState:
    continents: list[str]
    channels: list[str]
    payment_methods: list[str]
    pc_makes: list[str]
    date_min: pd.Timestamp | None
    date_max: pd.Timestamp | None


def apply_filters(df: pd.DataFrame, state: FilterState) -> pd.DataFrame:
    out = df

    if state.continents and "Continent" in out.columns:
        out = out[out["Continent"].isin(state.continents)]

    if state.channels and "Channel" in out.columns:
        out = out[out["Channel"].isin(state.channels)]

    if state.payment_methods and "Payment Method" in out.columns:
        out = out[out["Payment Method"].isin(state.payment_methods)]

    if state.pc_makes and "PC Make" in out.columns:
        out = out[out["PC Make"].isin(state.pc_makes)]

    if "purchase_date" in out.columns and (state.date_min or state.date_max):
        mask = pd.Series(True, index=out.index)
        if state.date_min is not None:
            mask &= out["purchase_date"].notna() & (out["purchase_date"] >= state.date_min)
        if state.date_max is not None:
            mask &= out["purchase_date"].notna() & (out["purchase_date"] <= state.date_max)
        out = out[mask]

    return out


def sidebar_filters(df: pd.DataFrame) -> FilterState:
    st.sidebar.header("Filters")

    def _opts(col: str) -> list[str]:
        if col not in df.columns:
            return []
        vals = df[col].dropna().astype(str).unique().tolist()
        vals = [v for v in vals if v.lower() not in {"nan", "n/a", "none", ""}]
        return sorted(vals)

    continents = st.sidebar.multiselect("Continent", options=_opts("Continent"))
    channels = st.sidebar.multiselect("Channel", options=_opts("Channel"))
    payment_methods = st.sidebar.multiselect("Payment Method", options=_opts("Payment Method"))
    pc_makes = st.sidebar.multiselect("PC Make", options=_opts("PC Make"))

    date_min: pd.Timestamp | None = None
    date_max: pd.Timestamp | None = None
    if "purchase_date" in df.columns:
        valid_dates = df["purchase_date"].dropna()
        if len(valid_dates) > 0:
            dmin = valid_dates.min().date()
            dmax = valid_dates.max().date()
            picked = st.sidebar.date_input("Purchase date range", value=(dmin, dmax))
            if isinstance(picked, tuple) and len(picked) == 2:
                date_min = pd.Timestamp(picked[0])
                date_max = pd.Timestamp(picked[1])

    return FilterState(
        continents=continents,
        channels=channels,
        payment_methods=payment_methods,
        pc_makes=pc_makes,
        date_min=date_min,
        date_max=date_max,
    )


def kpi(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def safe_group_sum(df: pd.DataFrame, by: str, value: str) -> pd.DataFrame:
    if by not in df.columns or value not in df.columns:
        return pd.DataFrame(columns=[by, value])
    tmp = df[[by, value]].dropna(subset=[by])
    tmp[value] = pd.to_numeric(tmp[value], errors="coerce")
    tmp = tmp.dropna(subset=[value])
    return tmp.groupby(by, as_index=False)[value].sum().sort_values(value, ascending=False)


def safe_group_mean(df: pd.DataFrame, by: str, value: str) -> pd.DataFrame:
    if by not in df.columns or value not in df.columns:
        return pd.DataFrame(columns=[by, value])
    tmp = df[[by, value]].dropna(subset=[by])
    tmp[value] = pd.to_numeric(tmp[value], errors="coerce")
    tmp = tmp.dropna(subset=[value])
    return tmp.groupby(by, as_index=False)[value].mean().sort_values(value, ascending=False)


def month_bucket(df: pd.DataFrame, date_col: str = "purchase_date") -> pd.Series:
    if date_col not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    return df[date_col].dt.to_period("M").dt.to_timestamp()


def ensure_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()
