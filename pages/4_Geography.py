from __future__ import annotations

import streamlit as st

from data_utils import DATA_FILE, apply_filters, load_raw, prep_data, safe_group_sum, sidebar_filters


def main() -> None:
    st.set_page_config(page_title="Geography", layout="wide")
    st.title("Geography")

    raw = load_raw(DATA_FILE)
    df = prep_data(raw)

    state = sidebar_filters(df)
    f = apply_filters(df, state)

    st.caption(f"Filtered rows: {len(f):,} / {len(df):,}")

    st.subheader("Net sales by continent")
    by_cont = safe_group_sum(f, "Continent", "net_sale")
    if len(by_cont):
        st.bar_chart(by_cont.set_index("Continent")["net_sale"], height=320)
        st.dataframe(by_cont, use_container_width=True, hide_index=True)
    else:
        st.info("No continent data available.")

    left, right = st.columns(2)

    with left:
        st.subheader("Top countries/states")
        by_country = safe_group_sum(f, "Country or State", "net_sale").head(25)
        if len(by_country):
            st.bar_chart(by_country.set_index("Country or State")["net_sale"], height=360)
            st.dataframe(by_country, use_container_width=True, hide_index=True)
        else:
            st.info("No country/state data available.")

    with right:
        st.subheader("Top provinces/cities")
        by_city = safe_group_sum(f, "Province or City", "net_sale").head(25)
        if len(by_city):
            st.bar_chart(by_city.set_index("Province or City")["net_sale"], height=360)
            st.dataframe(by_city, use_container_width=True, hide_index=True)
        else:
            st.info("No province/city data available.")


if __name__ == "__main__":
    main()
