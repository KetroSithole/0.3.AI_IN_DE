# PC Sales Streamlit Dashboard

A multi-page Streamlit dashboard for PC sales analytics (KPIs, trends, product/geography insights, data quality, segmentation, and visual insights).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

- The app looks for a CSV in this order:
  1. `pc_data.csv` (recommended for local use; **ignored by git** because it contains PII)
  2. `pc_data_public.csv` (sanitized sample used for GitHub/Streamlit Cloud)

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud, create a new app and select this repo/branch.
3. Set the **Main file path** to `app.py`.
4. Deploy.
