# 🌱 Climate-Driven Carbon Emission Automation Pipeline

Portofolio Modern Data Stack: **Prefect** (orkestrasi) + **DuckDB** (compute) +
**dbt** (transformasi & data quality) + **Prophet & scikit-learn** (forecasting
& deteksi anomali) + **Hugging Face Datasets** (data lake) + **GitHub Actions**
(CI/CD terjadwal) + **Google Sheets & Looker Studio** (reporting) + **Phi-3
SLM di Google Colab** (natural language analyst, Text-to-SQL).

## Arsitektur

```
GitHub Actions (cron harian / manual trigger)
        │
        ▼
pipeline-prefect.py
  ├─ Tarik data suhu (Open-Meteo API)
  ├─ Skoring anomali (IsolationForest, dilatih ulang tiap 7 hari)
  ├─ Forecast 30 hari (Prophet)
  ├─ Sync ke Data Lake (Hugging Face Datasets, parquet)
  ├─ Transformasi & data quality (dbt + DuckDB)
  └─ Reporting (Google Sheets → Looker Studio)

Streamlit Community Cloud (dashboard + trigger + AI chat)
  ├─ Baca data lake HF langsung (read-only, ringan)
  ├─ Tombol trigger workflow_dispatch ke GitHub Actions
  └─ Chat ke SLM (Phi-3-mini di Google Colab, via Cloudflare Tunnel)

Google Colab (GPU T4, SLM server)
  └─ Text-to-SQL + analisis natural language (colab_slm_server.py)
```

## Struktur repo

```
.
├── app.py                          # Dashboard Streamlit (deploy ke Streamlit Cloud)
├── pipeline-prefect.py             # Pipeline utama (dijalankan GitHub Actions)
├── colab_slm_server.py             # Server SLM (dijalankan manual di Google Colab)
├── requirements.txt
├── .github/workflows/run_pipeline.yml
└── src/climate_dbt/                # Project dbt
    ├── dbt_project.yml
    ├── profiles.yml
    ├── models/
    │   ├── sources.yml
    │   ├── schema.yml
    │   └── stg_carbon_emissions.sql
    └── tests/
        ├── assert_carbon_emission_non_negative.sql
        └── assert_temperature_reasonable_range.sql
```

Lihat `DEPLOY_STREAMLIT_CLOUD.md` untuk panduan deploy lengkap.
