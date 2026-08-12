import os
import subprocess
import requests
import random
import pandas as pd
import duckdb
import gspread
import json
import numpy as np
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
from prefect import task, flow
from huggingface_hub import HfApi, hf_hub_download

# ==========================================
# 🧠 IMPORT LIBRARY MACHINE LEARNING (DIBUNGKUS TRY/EXCEPT)
# ==========================================
# ⚠️ CATATAN: prophet, scikit-learn, xgboost, joblib HARUS ditambahkan ke requirements.txt
# environment Prefect Anda (HF Space / GitHub Actions / dll). Dibungkus try/except supaya
# kalau salah satu belum ter-install, pipeline INTI (tarik data + Sheets) tetap jalan --
# modul ML akan otomatis dilewati dengan peringatan, bukan meng-crash seluruh pipeline.
try:
    import joblib
    from sklearn.ensemble import IsolationForest
    ML_LIBS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Library ML (scikit-learn/joblib) belum ter-install: {e}. Modul anomali & carbon cap akan dilewati.")
    ML_LIBS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Library 'prophet' belum ter-install: {e}. Modul forecasting 30 hari akan dilewati.")
    PROPHET_AVAILABLE = False

# ==========================================
# ⚡ FIX PREFECT TIMEOUT & SQLITE LOCK (HUGGING FACE CLOUD OPTIMIZATION)
# ==========================================
os.environ["PREFECT_CLIENT_TIMEOUT"] = "120.0"
os.environ["PREFECT_API_DATABASE_CONNECTION_TIMEOUT"] = "60.0"
os.environ["PREFECT_API_DATABASE_TIMEOUT"] = "60.0"

# ==========================================
# CONFIGURATION & REGION COORDINATES
# ==========================================
REGIONS = {
    "Texas": {"lat": 31.9686, "lon": -99.9018, "base_min": 50.0, "base_max": 130.0},
    "Jakarta": {"lat": -6.2088, "lon": 106.8456, "base_min": 60.0, "base_max": 140.0}
}

HF_REPO_ID = "sigit48/carbon-emission-datalake"
HF_REMOTE_PATH = "data/carbon_emission_master.parquet"

# 🧠 Konfigurasi model ML (disimpan di repo HF yang sama, folder terpisah "models/")
HF_ISO_MODEL_PATH = "models/isolation_forest.joblib"
HF_ML_METADATA_PATH = "models/ml_metadata.json"
MIN_ROWS_FOR_ML_TRAINING = 30       # minimal baris histori aktual sebelum model dilatih
RETRAIN_INTERVAL_DAYS = 7           # latih ulang model tiap 7 hari (bukan tiap run!)
PROPHET_FORECAST_DAYS = 30
SHEETS_HISTORY_DAYS = 90            # Google Sheets hanya menyimpan N hari terakhir (histori penuh tetap di HF)


# ==========================================
# 1. TASK: EXTRACT TEMPERATURE (LIVE / HISTORICAL)
# ==========================================
@task(log_prints=True)
def get_temperature(lat, lon, target_date=None):
    if target_date and target_date != datetime.now().strftime("%Y-%m-%d"):
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={target_date}&end_date={target_date}&daily=temperature_2m_max&timezone=auto"
        try:
            res = requests.get(url).json()
            return res['daily']['temperature_2m_max'][0]
        except Exception:
            return 32.0 if lat == -6.2088 else 28.0
    else:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        try:
            res = requests.get(url).json()
            return res['current_weather']['temperature']
        except Exception:
            return 32.0 if lat == -6.2088 else 28.0


# ==========================================
# 2. TASK: GENERATE BATCH HARIAN (HANYA DATA AKTUAL HARI INI)
# ==========================================
# ⚠️ PERBAIKAN PENTING (riwayat):
# Versi lama meng-generate ULANG seluruh rentang tanggal dengan angka random baru di
# SETIAP run, menyebabkan histori berubah-ubah. Versi ini hanya membuat data untuk
# HARI INI. Prediksi 3-hari naif yang dulu ada di sini SUDAH DIPINDAH & DIGANTI oleh
# generate_prophet_forecast() (di bawah) yang memakai model Prophet asli berbasis
# seluruh histori, bukan rumus random.uniform() dummy.
@task(log_prints=True)
def generate_daily_batch():
    base_time = datetime.now()
    today_str = base_time.strftime("%Y-%m-%d")
    all_data = []

    print(f"📊 Menarik data aktual untuk hari ini ({today_str}) untuk Texas & Jakarta...")

    for region_name, config in REGIONS.items():
        temp = get_temperature(config['lat'], config['lon'], target_date=today_str)

        for j in range(5):
            base_em = round(random.uniform(config['base_min'], config['base_max']), 3)
            timestamp_str = base_time.strftime(f"%Y-%m-%d {10+j}:17:38")

            if region_name == "Jakarta":
                multiplier = random.uniform(1.10, 1.25) if temp > 32 else random.uniform(0.95, 1.05)
            else:
                multiplier = random.uniform(1.05, 1.15) if temp > 28 else random.uniform(0.95, 1.05)

            forecast_em = round(base_em * multiplier, 3)

            all_data.append({
                "generated_date": today_str,
                "generated_at": timestamp_str,
                "region": region_name,
                "live_temperature": float(temp),
                "base_emission_mt": float(base_em),
                "carbon_emission_forecast_mt": float(forecast_em)
            })

    df_today = pd.DataFrame(all_data)
    print(f"✓ Batch harian siap: {len(df_today)} baris data aktual hari ini.")
    return df_today


# ==========================================
# 2B. TASK: LATIH ATAU MUAT MODEL ANOMALY DETECTION
# ==========================================
# 🧠 DESAIN PENTING: model TIDAK di-fit ulang setiap run (itu akan membuat label
# anomali pada data HISTORIS ikut berubah-ubah setiap pipeline jalan -- bug yang sama
# seperti histori emisi yang goyang di awal proyek ini). Sebagai gantinya:
#   - Model dilatih SEKALI, lalu disimpan (joblib) ke HF di folder "models/".
#   - Run berikutnya cukup MEMUAT model yang sudah ada dan pakai .predict() saja.
#   - Model dilatih ULANG hanya setiap RETRAIN_INTERVAL_DAYS hari (default 7).
# ⚠️ CATATAN: XGBoost sudah TIDAK dipakai lagi di sini. recommended_carbon_cap_mt
# sekarang dihitung lewat FORMULA (lihat apply_ml_scoring), bukan regresi ML --
# lebih simpel & lebih mudah dijelaskan di portofolio karena aturannya eksplisit:
#   - Normal   : cap = carbon_emission_forecast_mt * 0.95 (pemotongan target 5%)
#   - Anomaly  : cap = base_emission_mt * 1.05 (toleransi buffer sementara 5%)
@task(log_prints=True)
def train_or_load_ml_models(df_actual_history):
    if not ML_LIBS_AVAILABLE:
        print("⚠️ Library ML tidak tersedia, modul deteksi anomali dilewati.")
        return None

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN tidak ditemukan, modul deteksi anomali dilewati.")
        return None

    today_str = datetime.now().strftime("%Y-%m-%d")

    should_retrain = True
    try:
        meta_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ML_METADATA_PATH, repo_type="dataset", token=hf_token)
        with open(meta_path) as f:
            metadata = json.load(f)
        last_trained = datetime.strptime(metadata["last_trained_date"], "%Y-%m-%d")
        days_since_training = (datetime.now() - last_trained).days
        if days_since_training < RETRAIN_INTERVAL_DAYS:
            should_retrain = False
            print(f"ℹ️ Model terakhir dilatih {days_since_training} hari lalu (< {RETRAIN_INTERVAL_DAYS} hari) -> muat model lama, tidak dilatih ulang.")
    except Exception:
        print("ℹ️ Belum ada model tersimpan (inisialisasi pertama) -> akan dilatih.")

    if not should_retrain:
        try:
            iso_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ISO_MODEL_PATH, repo_type="dataset", token=hf_token)
            iso_model = joblib.load(iso_path)
            print("✓ Model anomali berhasil dimuat dari HF (tidak dilatih ulang).")
            return iso_model
        except Exception as e:
            print(f"⚠️ Gagal memuat model lama ({str(e)}), akan melatih ulang.")

    # --- LATIH MODEL BARU ---
    df_train = df_actual_history.dropna(subset=["live_temperature", "base_emission_mt"])
    if len(df_train) < MIN_ROWS_FOR_ML_TRAINING:
        print(f"⚠️ Data histori aktual baru {len(df_train)} baris (< {MIN_ROWS_FOR_ML_TRAINING}), "
              f"belum cukup untuk melatih model anomali. Dilewati untuk sementara.")
        return None

    print(f"🧠 Melatih ulang model IsolationForest dari {len(df_train)} baris histori aktual...")

    iso_model = IsolationForest(contamination=0.05, random_state=42)
    iso_model.fit(df_train[["live_temperature", "base_emission_mt"]])

    local_iso_path, local_meta_path = "temp_iso.joblib", "temp_meta.json"
    joblib.dump(iso_model, local_iso_path)
    with open(local_meta_path, "w") as f:
        json.dump({"last_trained_date": today_str, "n_training_rows": len(df_train)}, f)

    try:
        api = HfApi()
        for local_path, remote_path in [
            (local_iso_path, HF_ISO_MODEL_PATH),
            (local_meta_path, HF_ML_METADATA_PATH),
        ]:
            api.upload_file(path_or_fileobj=local_path, path_in_repo=remote_path, repo_id=HF_REPO_ID, repo_type="dataset", token=hf_token)
        print(f"🚀 Model anomali baru berhasil disimpan ke HF ({HF_ISO_MODEL_PATH}).")
    except Exception as e:
        print(f"⚠️ Gagal upload model ke HF (tidak fatal, model tetap dipakai untuk run ini): {str(e)}")
    finally:
        for p in [local_iso_path, local_meta_path]:
            if os.path.exists(p):
                os.remove(p)

    return iso_model


# ==========================================
# 2C. TASK: SKOR BATCH HARI INI (ANOMALY FLAG 0/1 + CARBON CAP BERBASIS FORMULA)
# ==========================================
# 🎯 PENTING: task ini HANYA melakukan .predict() pada batch BARU (bukan fit ulang),
# jadi is_anomaly pada baris historis TIDAK PERNAH berubah setelah pertama kali
# ditulis. is_anomaly sekarang berupa INTEGER: 0 = Normal, 1 = Anomaly, NULL kalau
# model belum cukup data untuk dilatih (bukan 3 nilai string seperti sebelumnya).
@task(log_prints=True)
def apply_ml_scoring(df_today_batch, iso_model):
    if iso_model is None:
        print("ℹ️ Model anomali belum tersedia (data historis belum cukup). is_anomaly & "
              "recommended_carbon_cap_mt diisi NULL untuk batch ini.")
        df_today_batch["is_anomaly"] = pd.array([None] * len(df_today_batch), dtype="Int64")
        df_today_batch["recommended_carbon_cap_mt"] = np.nan
        return df_today_batch

    X_anomaly = df_today_batch[["live_temperature", "base_emission_mt"]]
    anomaly_codes = iso_model.predict(X_anomaly)  # sklearn: -1 = anomaly, 1 = normal
    df_today_batch["is_anomaly"] = pd.array([1 if c == -1 else 0 for c in anomaly_codes], dtype="Int64")

    # 🍃 Carbon cap sekarang FORMULA (bukan regresi ML), berdasarkan status anomali:
    #   - Normal (0)  -> cap = carbon_emission_forecast_mt * 0.95 (target pemotongan 5%,
    #                    mendekati standar SBTi ~4.2%/tahun untuk Scope 1&2)
    #   - Anomaly (1) -> cap = base_emission_mt * 1.05 (buffer toleransi sementara 5%,
    #                    supaya tidak langsung dianggap melanggar batas saat lonjakan terdeteksi)
    df_today_batch["recommended_carbon_cap_mt"] = np.where(
        df_today_batch["is_anomaly"] == 1,
        df_today_batch["base_emission_mt"] * 1.05,
        df_today_batch["carbon_emission_forecast_mt"] * 0.95,
    ).round(3)

    n_anomaly = (df_today_batch["is_anomaly"] == 1).sum()
    print(f"✓ Skoring anomali selesai. {n_anomaly} dari {len(df_today_batch)} baris hari ini terdeteksi sebagai anomali.")
    return df_today_batch


# ==========================================
# 2D. TASK: FORECAST 30 HARI DENGAN PROPHET (GANTI PREDIKSI NAIF LAMA)
# ==========================================
# 🔮 Prophet di-fit ULANG setiap run memakai SELURUH histori aktual yang tersedia --
# ini MEMANG DISENGAJA (beda dengan model anomali/cap di atas), karena forecast
# memang harus selalu pakai data terbaru. Region output tetap pakai label
# "<Wilayah> (Prediction)" SUPAYA KOMPATIBEL dengan seluruh sistem yang sudah ada
# (dashboard Looker Studio, prompt SLM, fungsi build_comparison_insight di Colab).
@task(log_prints=True)
def generate_prophet_forecast(df_actual_history):
    if not PROPHET_AVAILABLE:
        print("⚠️ Library 'prophet' tidak tersedia, forecast 30 hari dilewati.")
        return pd.DataFrame()

    all_forecasts = []

    for region_name in REGIONS.keys():
        df_region = df_actual_history[df_actual_history["region"] == region_name].copy()
        if df_region.empty:
            continue

        # Agregasi ke rata-rata HARIAN dulu (data mentah punya 5 baris/hari) -- Prophet
        # butuh satu nilai 'y' per tanggal 'ds', bukan beberapa nilai duplikat per hari.
        df_daily = df_region.groupby("generated_date").agg(
            y=("carbon_emission_forecast_mt", "mean")
        ).reset_index()
        df_daily.columns = ["ds", "y"]
        df_daily["ds"] = pd.to_datetime(df_daily["ds"])

        if len(df_daily) < 5:
            print(f"⚠️ Histori harian {region_name} baru {len(df_daily)} hari (< 5), forecast Prophet dilewati untuk wilayah ini.")
            continue

        try:
            model = Prophet(daily_seasonality=False, weekly_seasonality=True, interval_width=0.95)
            model.fit(df_daily)

            future = model.make_future_dataframe(periods=PROPHET_FORECAST_DAYS)
            forecast = model.predict(future)

            # Ambil HANYA tanggal masa depan (bukan tanggal historis yang ikut ke-forecast oleh Prophet)
            last_actual_date = df_daily["ds"].max()
            forecast_future = forecast[forecast["ds"] > last_actual_date]

            for _, row in forecast_future.iterrows():
                all_forecasts.append({
                    "generated_date": row["ds"].strftime("%Y-%m-%d"),
                    "generated_at": row["ds"].strftime("%Y-%m-%d 00:00:00"),
                    "region": f"{region_name} (Prediction)",
                    "live_temperature": np.nan,
                    "base_emission_mt": np.nan,
                    "carbon_emission_forecast_mt": round(row["yhat"], 3),
                    "prophet_upper": round(row["yhat_upper"], 3),
                    "prophet_lower": round(row["yhat_lower"], 3),
                })
            print(f"✓ Forecast Prophet {PROPHET_FORECAST_DAYS} hari untuk {region_name} berhasil dibuat.")
        except Exception as e:
            print(f"⚠️ Gagal membuat forecast Prophet untuk {region_name}: {str(e)}")

    return pd.DataFrame(all_forecasts)


# ==========================================
# 3. TASK: SINKRONISASI KE HUGGING FACE DATA LAKE (PARQUET) — SUMBER KEBENARAN
# ==========================================
# Task ini mengunduh master parquet lama, menggabungkannya dengan batch baru,
# lalu dedupe berdasarkan (generated_at, region):
#   - Baris "actual" lama TIDAK PERNAH tertimpa, karena generated_at-nya unik per hari
#     dan batch baru hanya berisi HARI INI.
#   - Baris "prediction" untuk tanggal yang sama SENGAJA di-refresh (keep="last"),
#     supaya proyeksinya selalu pakai model terbaru.
# Hasil gabungan (df_combined) inilah yang dipakai untuk mengisi DuckDB, dbt, dan Sheets,
# supaya semua layer selalu konsisten dengan satu sumber data yang sama.
@task(log_prints=True)
def sync_with_datalake(df_new_batch):
    print("🎯 Menyingkronkan batch baru dengan Hugging Face Data Lake (Parquet)...")

    hf_token = os.environ.get("HF_TOKEN")
    local_parquet_path = "temp_data.parquet"

    if not hf_token:
        print("⚠️ Token 'HF_TOKEN' tidak ditemukan. Melewati sinkronisasi Data Lake, memakai batch baru saja.")
        return df_new_batch

    try:
        print("Mengunduh data master lama dari Hugging Face...")
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_REMOTE_PATH,
            repo_type="dataset",
            token=hf_token
        )
        df_old = pd.read_parquet(downloaded_path)
        print(f"✓ Data lama ditemukan ({len(df_old)} baris). Menggabungkan dengan batch baru...")
        df_combined = pd.concat([df_old, df_new_batch], ignore_index=True)
    except Exception as e:
        print(f"ℹ️ Belum ada data lake lama di repo (Inisialisasi Pertama): {str(e)}")
        df_combined = df_new_batch

    df_combined = df_combined.drop_duplicates(subset=["generated_at", "region"], keep="last")
    df_combined = df_combined.sort_values("generated_at").reset_index(drop=True)

    # 🔧 MIGRASI SKEMA: versi lama menyimpan is_anomaly sebagai STRING ('Normal',
    # 'Anomalous Spike', 'Insufficient Data'). Skema baru pakai INTEGER (0/1/NULL).
    # Baris lama yang masih string dikonversi paksa ke NULL di sini -- SEKALI dan
    # PERMANEN, karena hasilnya diupload balik ke HF -- supaya run berikutnya sudah
    # bersih dan tidak perlu migrasi berulang.
    if "is_anomaly" in df_combined.columns:
        n_before = df_combined["is_anomaly"].apply(lambda x: isinstance(x, str)).sum()
        df_combined["is_anomaly"] = pd.to_numeric(df_combined["is_anomaly"], errors="coerce")
        if n_before > 0:
            print(f"🔧 Migrasi skema: {n_before} baris is_anomaly lama (format string) dikonversi ke NULL.")

    df_combined.to_parquet(local_parquet_path, index=False)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_parquet_path,
            path_in_repo=HF_REMOTE_PATH,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_token
        )
        print(f"🚀 Sukses! Data Lake Parquet diperbarui. Total master data sekarang: {len(df_combined)} baris.")
    except Exception as upload_error:
        print(f"❌ Gagal mengunggah file ke HF Datasets: {str(upload_error)}")
    finally:
        if os.path.exists(local_parquet_path):
            os.remove(local_parquet_path)

    return df_combined


# ==========================================
# 3A. TASK: AMBIL HISTORI AKTUAL (TANPA BARIS PREDIKSI) UNTUK TRAINING ML & PROPHET
# ==========================================
@task(log_prints=True)
def get_historical_actual_data():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return pd.DataFrame()
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_REMOTE_PATH, repo_type="dataset", token=hf_token)
        df_master = pd.read_parquet(path)
        df_actual = df_master[~df_master["region"].str.contains("Prediction", na=False)].copy()
        print(f"✓ Histori aktual ditemukan: {len(df_actual)} baris (dipakai untuk training ML & Prophet).")
        return df_actual
    except Exception as e:
        print(f"ℹ️ Belum ada data historis di HF untuk training ML/Prophet (inisialisasi pertama): {str(e)}")
        return pd.DataFrame()


# ==========================================
# 3B. TASK: PASTIKAN README.md DATASET PUNYA METADATA (AKTIFKAN DATASET VIEWER)
# ==========================================
# ⚠️ CATATAN: "Dataset viewer is not available" di halaman HF BUKAN error yang
# merusak fungsi apa pun -- hf_hub_download tetap bisa mengambil file parquet dengan
# normal. Itu hanya terjadi karena README.md dataset kosong, jadi HF tidak tahu di
# mana letak file datanya untuk di-preview. Task ini menambahkan metadata YAML supaya
# Dataset Viewer di web HF ikut aktif (murni kosmetik/dokumentasi, aman dijalankan
# berulang kali / idempotent).
@task(log_prints=True)
def ensure_dataset_readme():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️ HF_TOKEN tidak ditemukan, melewati update README.md dataset.")
        return

    readme_content = f"""---
configs:
- config_name: default
  data_files:
  - split: train
    path: {HF_REMOTE_PATH}
---

# Carbon Emission Data Lake

Dataset ini berisi data historis & prediksi emisi karbon serta suhu untuk wilayah
Texas dan Jakarta, dihasilkan secara otomatis oleh pipeline Prefect + dbt.

Kolom: `generated_date`, `generated_at`, `region`, `live_temperature`,
`base_emission_mt`, `carbon_emission_forecast_mt`, `prophet_upper`, `prophet_lower`,
`is_anomaly` (0=Normal, 1=Anomaly, NULL=model belum cukup data), `recommended_carbon_cap_mt`.

Modul ML: Prophet (forecast 30 hari), IsolationForest (deteksi anomali). Carbon cap
dihitung via formula berbasis status anomali: Normal -> forecast x0.95, Anomaly ->
baseline x1.05 (buffer toleransi sementara).
"""
    local_readme_path = "temp_README.md"
    with open(local_readme_path, "w") as f:
        f.write(readme_content)

    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_readme_path,
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            token=hf_token
        )
        print("✓ README.md dataset diperbarui, Dataset Viewer di HF seharusnya aktif dalam beberapa saat.")
    except Exception as e:
        print(f"⚠️ Gagal memperbarui README.md dataset (tidak fatal, pipeline tetap lanjut): {str(e)}")
    finally:
        if os.path.exists(local_readme_path):
            os.remove(local_readme_path)


# ==========================================
# 4. TASK: LOAD DATA (MASTER LENGKAP) KE DUCKDB
# ==========================================
@task(log_prints=True)
def load_to_duckdb(df_master):
    # Pastikan semua kolom yang diharapkan selalu ada, meski salah satu modul ML
    # (Prophet / IsolationForest / XGBoost) belum sempat menghasilkan data di run
    # pertama -- supaya INSERT tidak gagal karena mismatch kolom.
    expected_cols = [
        "generated_date", "generated_at", "region", "live_temperature",
        "base_emission_mt", "carbon_emission_forecast_mt",
        "prophet_upper", "prophet_lower", "is_anomaly", "recommended_carbon_cap_mt",
    ]
    for col in expected_cols:
        if col not in df_master.columns:
            df_master[col] = np.nan
    df_master = df_master[expected_cols]

    # 🔧 Pengaman kedua: pastikan is_anomaly numerik sebelum di-INSERT ke DuckDB.
    # Migrasi utama sudah dilakukan di sync_with_datalake, tapi ini jaga-jaga kalau
    # ada jalur data lain (mis. df_master dari sumber selain sync_with_datalake) yang
    # belum sempat termigrasi -- errors="coerce" mengubah string tak dikenal jadi NULL
    # alih-alih meng-crash seluruh pipeline seperti sebelumnya.
    df_master["is_anomaly"] = pd.to_numeric(df_master["is_anomaly"], errors="coerce")

    conn = duckdb.connect('your_project.db')
    conn.execute("DROP TABLE IF EXISTS raw_carbon_emissions")
    conn.execute("""
        CREATE TABLE raw_carbon_emissions (
            generated_date VARCHAR,
            generated_at VARCHAR,
            region VARCHAR,
            live_temperature DOUBLE,
            base_emission_mt DOUBLE,
            carbon_emission_forecast_mt DOUBLE,
            prophet_upper DOUBLE,
            prophet_lower DOUBLE,
            is_anomaly INTEGER,
            recommended_carbon_cap_mt DOUBLE
        )
    """)
    conn.execute("INSERT INTO raw_carbon_emissions SELECT * FROM df_master")
    total_rows = conn.execute("SELECT COUNT(*) FROM raw_carbon_emissions").fetchone()[0]
    print(f"✓ Data terkonsolidasi masuk ke DuckDB. Total baris: {total_rows}")
    conn.close()


# ==========================================
# 5. TASK: JALANKAN TRANSFORMASI dbt & QUALITY CHECKS
# ==========================================
@task(log_prints=True)
def run_dbt_transformation():
    print("Memulai proses transformasi dbt...")
    dbt_cwd = os.path.abspath("./src/climate_dbt")

    if not os.path.exists(dbt_cwd):
        raise FileNotFoundError(f"Folder proyek dbt 'climate_dbt' tidak ditemukan di {dbt_cwd}")

    run_result = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd=dbt_cwd,
        capture_output=True,
        text=True
    )
    if run_result.returncode != 0:
        # ⚠️ PERBAIKAN: dbt sering menaruh detail error di stdout, bukan cuma stderr.
        # Cetak keduanya supaya diagnosis tidak kehilangan info penting.
        print(run_result.stdout)
        print(run_result.stderr)
        raise RuntimeError("Gagal menjalankan dbt run.")

    print("✓ dbt run sukses! Menjalankan dbt test...")
    test_result = subprocess.run(
        ["dbt", "test", "--profiles-dir", "."],
        cwd=dbt_cwd,
        capture_output=True,
        text=True
    )
    if test_result.returncode != 0:
        # ⚠️ PERBAIKAN: sebelumnya cuma print stderr, padahal detail test mana yang
        # gagal (nama test, jumlah baris melanggar) biasanya ada di stdout dbt.
        print(test_result.stdout)
        print(test_result.stderr)
        raise RuntimeError("❌ Data Quality Check Gagal pada dbt test.")
    print("✓ dbt test sukses! Kualitas data bersih terjamin.")


# ==========================================
# 6. TASK: LOAD CLEAN DATA TO GOOGLE SHEETS (IDEMPOTENT OVERWRITE)
# ==========================================
@task(log_prints=True)
def load_clean_data_to_sheets(spreadsheet_name):
    conn = duckdb.connect('your_project.db')
    df_clean = conn.execute("SELECT * FROM stg_carbon_emissions").df()
    conn.close()

    # ⚠️ PERBAIKAN: sebelumnya SELURUH histori (sejak 18 Juni) ditulis ulang ke Sheets
    # di SETIAP run -- makin lama makin besar, padahal Sheets/Looker Studio biasanya
    # cuma butuh tren terbaru untuk reporting. Histori LENGKAP tetap aman tersimpan
    # di parquet HF (sumber kebenaran); Sheets sekarang dibatasi ke SHEETS_HISTORY_DAYS
    # hari terakhir saja. Baris prediksi (tanggal di masa depan) otomatis tetap ikut
    # karena tanggalnya pasti >= cutoff.
    if "event_date" in df_clean.columns:
        df_clean["event_date"] = pd.to_datetime(df_clean["event_date"])
        cutoff = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=SHEETS_HISTORY_DAYS)
        n_before = len(df_clean)
        df_clean = df_clean[df_clean["event_date"] >= cutoff].copy()
        print(f"📏 Sheets dibatasi ke {SHEETS_HISTORY_DAYS} hari terakhir: {n_before} -> {len(df_clean)} baris "
              f"(histori lengkap tetap tersimpan di data lake HF).")

    for col in df_clean.columns:
        if pd.api.types.is_datetime64_any_dtype(df_clean[col]) or df_clean[col].dtype == 'object':
            try:
                df_clean[col] = df_clean[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except AttributeError:
                df_clean[col] = df_clean[col].astype(str)

    # ⚠️ PERBAIKAN PENTING: kolom numerik untuk baris prediksi Prophet (live_temperature,
    # base_emission_mt, dll) sengaja diisi NaN. JSON standar TIDAK mengenal NaN/Infinity
    # (beda dengan Python), jadi requests.post ke Google Sheets API gagal total dengan
    # "Out of range float values are not JSON compliant" sebelum sempat terkirim.
    # np.nan/np.inf harus diganti None (jadi JSON null) di SEMUA kolom, bukan cuma
    # kolom datetime/object yang sudah ditangani di atas.
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    df_clean = df_clean.astype(object).where(pd.notnull(df_clean), None)

    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    secret_creds = os.environ.get("GOOGLE_CREDENTIALS")

    if not secret_creds:
        raise ValueError("Secret 'GOOGLE_CREDENTIALS' tidak ditemukan!")

    creds_dict = json.loads(secret_creds)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open(spreadsheet_name).sheet1

    print(f"Mengosongkan Google Sheets untuk memuat {SHEETS_HISTORY_DAYS} hari terakhir + proyeksi masa depan...")
    sheet.clear()

    header = df_clean.columns.values.tolist()
    rows = df_clean.values.tolist()
    payload = [header] + rows

    sheet.append_rows(payload)
    print(f"✓ Sukses! Google Sheets ter-overwrite bersih dengan {len(df_clean)} baris data.")


# ==========================================
# 6B. NOTIFIKASI KEGAGALAN (WEBHOOK, OPSIONAL)
# ==========================================
# 🔔 Kirim notifikasi kalau pipeline gagal di tahap mana pun. Format payload kompatibel
# dengan Slack Incoming Webhook (juga otomatis kompatibel dengan Discord kalau URL
# webhook Discord-nya ditambah akhiran "/slack"). Sepenuhnya OPSIONAL -- kalau
# NOTIFY_WEBHOOK_URL tidak di-set, fungsi ini diam saja tanpa mengganggu pipeline.
def notify_failure(error_message: str):
    webhook_url = os.environ.get("NOTIFY_WEBHOOK_URL")
    if not webhook_url:
        print("ℹ️ NOTIFY_WEBHOOK_URL tidak di-set, notifikasi kegagalan dilewati.")
        return
    try:
        payload = {
            "text": (
                f"🚨 *Carbon Emission Pipeline GAGAL*\n"
                f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Error: {error_message}"
            )
        }
        requests.post(webhook_url, json=payload, timeout=10)
        print("✓ Notifikasi kegagalan terkirim.")
    except Exception as notify_error:
        # Kegagalan kirim notifikasi TIDAK BOLEH menutupi error asli pipeline
        print(f"⚠️ Gagal mengirim notifikasi kegagalan (diabaikan): {notify_error}")


# ==========================================
# 7. MAIN PREFECT PIPELINE ORCHESTRATION
# ==========================================
@flow(name="Carbon Emission End-to-End Pipeline")
def carbon_etl_flow():
    NAMA_GOOGLE_SHEETS = "Carbon Emission Data Store"

    try:
        # 0. Ambil histori aktual (tanpa baris prediksi) -> dipakai untuk training ML & Prophet
        df_actual_history = get_historical_actual_data()

        # 1. Generate hanya batch data aktual hari ini
        df_today = generate_daily_batch()

        # 2. Latih/muat model ML (IsolationForest + XGBoost), lalu skor batch hari ini
        #    -> HANYA batch baru yang di-skor, histori lama tidak pernah disentuh ulang.
        iso_model = train_or_load_ml_models(df_actual_history)
        df_today_scored = apply_ml_scoring(df_today, iso_model)

        # 3. Forecast 30 hari ke depan dengan Prophet, berbasis histori + data hari ini
        df_history_updated = pd.concat([df_actual_history, df_today], ignore_index=True)
        df_prophet = generate_prophet_forecast(df_history_updated)

        # 4. Gabungkan batch hari ini (sudah diskor ML) + forecast Prophet, lalu sync ke HF
        df_new_batch = pd.concat([df_today_scored, df_prophet], ignore_index=True)
        df_master = sync_with_datalake(df_new_batch)

        # 4B. Pastikan README.md dataset punya metadata (aktifkan Dataset Viewer di HF)
        ensure_dataset_readme()

        # 5. Muat master lengkap ke DuckDB untuk transformasi & reporting
        load_to_duckdb(df_master)
        run_dbt_transformation()

        # 6. Sinkronisasi ke Reporting Layer (Sheets -> Looker Studio)
        load_clean_data_to_sheets(NAMA_GOOGLE_SHEETS)

    except Exception as e:
        # 🔔 Notifikasi dulu, BARU lempar ulang error-nya -- supaya Prefect (dan
        # subprocess di app.py) tetap tahu run ini gagal, tapi Anda juga dapat
        # pemberitahuan tanpa harus terus-menerus cek log secara manual.
        notify_failure(str(e))
        raise


if __name__ == "__main__":
    carbon_etl_flow()
