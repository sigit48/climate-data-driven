# ==========================================
# RUNNING SERVER SLM + DUCKDB API (COLAB SIDE)
# ==========================================
!pip install uvicorn fastapi transformers accelerate bitsandbytes nest_asyncio duckdb pandas requests huggingface_hub
import os

# ⚠️ PERBAIKAN OOM: mengurangi fragmentasi memori CUDA. HARUS di-set SEBELUM torch
# di-import / CUDA context dibuat, makanya diletakkan paling atas.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import gc
import nest_asyncio
import uvicorn
import torch
import duckdb
import pandas as pd
import time
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from huggingface_hub import hf_hub_download

app = FastAPI(title="Carbon ESG SLM Analyst API")
nest_asyncio.apply()

# ⚠️ Isi token HF di sini kalau repo dataset Anda PRIVATE (biarkan None kalau publik).
# Sebaiknya simpan lewat Colab Secrets (ikon kunci di sidebar kiri) lalu:
# from google.colab import userdata; HF_TOKEN = userdata.get('HF_TOKEN')
HF_TOKEN = os.environ.get("HF_TOKEN")  # atau isi string token langsung, mis. "hf_xxxxx"

HF_REPO_ID = "sigit48/carbon-emission-datalake"
HF_REMOTE_PATH = "data/carbon_emission_master.parquet"

# 1. Muat Model ke GPU
model_id = "microsoft/Phi-3-mini-4k-instruct"

# 🧪 TOGGLE UNTUK BENCHMARK: set True untuk MEMAKSA mode 8-bit quantized (tanpa perlu
# menunggu OOM beneran terjadi), supaya bisa dibandingkan apple-to-apple dengan fp16.
# Cara pakai: jalankan sekali dengan False (catat latency), restart runtime, jalankan
# lagi dengan True (catat latency), lalu bandingkan hasilnya.
FORCE_8BIT_QUANTIZATION = False

print("⏳ Memuat model Phi-3-Mini ke GPU T4...")

if not torch.cuda.is_available():
    print("⚠️ GPU tidak terdeteksi! Pastikan Runtime > Change runtime type > GPU (T4) sudah aktif.")

# ⚠️ PERBAIKAN OOM: kalau sel ini pernah dijalankan sebelumnya di sesi yang sama
# (misalnya setelah error dan Anda coba lagi TANPA restart runtime), model & pipeline
# lama masih nyangkut di VRAM. Bersihkan dulu sebelum load model baru, supaya tidak
# menumpuk 2x memori seperti yang menyebabkan "CUDA out of memory" di atas.
for var_name in ["model", "pipe", "tokenizer"]:
    if var_name in globals():
        del globals()[var_name]
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"ℹ️ VRAM tersedia sebelum load: {free_mem / 1e9:.2f} GB dari total {total_mem / 1e9:.2f} GB")

tokenizer = AutoTokenizer.from_pretrained(model_id)

if FORCE_8BIT_QUANTIZATION:
    print("🧪 FORCE_8BIT_QUANTIZATION=True -> memuat langsung dalam mode 8-bit (untuk benchmark).")
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    print("✓ Model dimuat dalam mode 8-bit quantized (dipaksa untuk benchmark).")
else:
    try:
        # Percobaan pertama: fp16 penuh (butuh ~7.6GB, seharusnya muat di T4)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map={"": 0} if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        print("✓ Model dimuat dalam mode fp16 penuh.")
    except torch.cuda.OutOfMemoryError:
        # ⚠️ FALLBACK: kalau VRAM tetap tidak cukup (mis. GPU dipakai proses lain / sesi
        # Colab lama), muat ulang dengan quantization 8-bit -- ukuran turun jadi ~4GB,
        # jauh lebih aman, dengan penurunan kualitas yang minimal untuk tugas Text-to-SQL ini.
        print("⚠️ OOM di mode fp16, mencoba ulang dengan quantization 8-bit (butuh VRAM lebih sedikit)...")
        gc.collect()
        torch.cuda.empty_cache()
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map={"": 0}
        )
        print("✓ Model dimuat dalam mode 8-bit quantized (fallback).")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("🎯 Model Berhasil Dimuat!")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"📊 VRAM terpakai setelah model dimuat: {allocated:.2f} GB")

# 2. Unduh Dataset LANGSUNG dari file yang di-upload pipeline-prefect.py
# ⚠️ PERBAIKAN: endpoint lama (/api/datasets/.../parquet/default/train) adalah endpoint
# AUTO-KONVERSI Hugging Face yang hanya aktif kalau bot refs/convert/parquet sudah
# memproses repo Anda -- ini TIDAK akan pernah terjadi untuk repo berisi file parquet
# manual seperti punya Anda, makanya selalu balas Status Code 400.
# Sekarang kita ambil file parquet-nya LANGSUNG lewat hf_hub_download, dengan repo_id
# & path yang sama persis dengan yang dipakai pipeline-prefect.py saat upload.
print(f"📥 Mengunduh {HF_REMOTE_PATH} langsung dari repo {HF_REPO_ID}...")

db_conn = None
try:
    downloaded_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_REMOTE_PATH,
        repo_type="dataset",
        token=HF_TOKEN
    )
    df_temp = pd.read_parquet(downloaded_path)

    db_conn = duckdb.connect()
    db_conn.execute("CREATE OR REPLACE TABLE raw_data AS SELECT * FROM df_temp")
    print(f"✅ Data Sukses Ter-registrasi di DuckDB! Total: {len(df_temp)} baris.")
except Exception as e:
    print(f"❌ Gagal memuat dataset: {str(e)}")
    print("   Pastikan pipeline-prefect.py sudah pernah dijalankan minimal 1x (supaya file")
    print(f"   {HF_REMOTE_PATH} sudah ada di repo {HF_REPO_ID}), dan HF_TOKEN sudah diisi kalau repo private.")


class QueryRequest(BaseModel):
    question: str


def generate_slm_response(messages, max_tokens=150):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=max_tokens, max_length=None, do_sample=False)
    return outputs[0]['generated_text'][len(prompt):].strip()


def clean_sql_string(raw_sql):
    # 🎯 PRIORITAS UTAMA: ekstrak dari dalam code fence ```sql ... ```
    # Model jauh lebih disiplin berhenti setelah penutup ``` karena pola ini sangat
    # kuat di data training markdown -> mengurangi risiko model "nyerocos" bikin
    # teks tambahan setelah SQL selesai.
    fence_match = re.search(r"```sql\s*(.*?)```", raw_sql, re.IGNORECASE | re.DOTALL)
    if fence_match:
        clean = fence_match.group(1).strip()
    else:
        # Fallback: model tidak memakai code fence, bersihkan dengan cara lama.
        clean = raw_sql.replace("```sql", "").replace("```", "").strip()
        for stop_marker in ["\nQuestion", " Question:", "\nSQL:", " SQL:", ";"]:
            idx = clean.find(stop_marker)
            if idx != -1:
                clean = clean[:idx]

    clean = clean.replace("\n", " ").strip().rstrip(";")
    match = re.search(r"(SELECT\s+.*)", clean, re.IGNORECASE)
    return match.group(1) if match else clean


# 📌 Kata kunci yang menandakan pertanyaan bersifat LOGIKA/SEBAB-AKIBAT
# (bukan sekadar "berapa nilai X"), misalnya "jika suhu naik apakah emisi juga naik?"
ANALYTICAL_KEYWORDS = [
    "apakah", "jika", "kalau", "pengaruh", "hubungan", "korelasi", "kenapa", "mengapa",
    "does", "if ", "correlat", "relationship", "affect", "why", "trend", "berpengaruh"
]


def is_analytical_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ANALYTICAL_KEYWORDS)


def find_column(df, must_contain):
    """Cari nama kolom yang mengandung substring tertentu (case-insensitive)."""
    for col in df.columns:
        if must_contain in col.lower():
            return col
    return None


# 🌐 DETEKSI BAHASA: jawaban sekarang mengikuti bahasa pertanyaan (satu bahasa saja,
# bukan format [ID]/[EN] dobel yang kaku). Default ke Indonesia (audiens utama aplikasi
# ini) kecuali pertanyaan jelas ditulis dalam bahasa Inggris.
_ID_MARKERS = [
    "apakah", "berapa", "bagaimana", "kenapa", "mengapa", "yang", "dengan", "dan",
    "atau", "adalah", "tidak", "emisi", "suhu", "hari", "minggu", "ini", "itu",
    "wilayah", "rata-rata", "tertinggi", "terendah", "solusi", "cara", "bandingkan",
    "jika", "kalau", "untuk", "dari", "pada", "ada", "batas", "anomali",
]
_EN_MARKERS = [
    "what", "how", "why", "the", "and", "compare", "average", "highest", "lowest",
    "please", "does", "anomaly", "today", "week", "is there", "recommend", "cap",
]


def detect_language(question: str) -> str:
    q = question.lower()
    id_hits = sum(1 for w in _ID_MARKERS if w in q)
    en_hits = sum(1 for w in _EN_MARKERS if w in q)
    return "en" if en_hits > id_hits else "id"


def humanize_metric(col: str, lang: str) -> str:
    """Ubah nama kolom snake_case (mis. avg_emission) jadi frasa natural."""
    col_l = col.lower()
    prefix_map = (
        {"avg_": "rata-rata ", "max_": "nilai maksimum ", "min_": "nilai minimum ", "sum_": "total "}
        if lang == "id" else
        {"avg_": "average ", "max_": "maximum ", "min_": "minimum ", "sum_": "total "}
    )
    subject_map = (
        {"temp": "suhu", "emission": "emisi karbon", "cap": "carbon cap", "anomaly": "status anomali"}
        if lang == "id" else
        {"temp": "temperature", "emission": "carbon emission", "cap": "carbon cap", "anomaly": "anomaly status"}
    )
    prefix, body = "", col_l
    for p, label in prefix_map.items():
        if col_l.startswith(p):
            prefix, body = label, col_l[len(p):]
            break
    subject = next((label for key, label in subject_map.items() if key in body), col.replace("_", " "))
    return f"{prefix}{subject}".strip()


def build_correlation_insight(df_result: pd.DataFrame, user_q: str, lang: str = "id"):
    """
    Menghitung korelasi antara suhu & emisi SECARA MATEMATIS (bukan diserahkan ke SLM,
    karena model sekecil Phi-3-mini gampang salah/halusinasi soal reasoning numerik).
    Mengembalikan None kalau kolom yang dibutuhkan tidak ditemukan.
    """
    temp_col = find_column(df_result, "temp")
    emission_col = find_column(df_result, "emission")

    if temp_col is None or emission_col is None or len(df_result) < 2:
        return None

    df_clean = df_result[[temp_col, emission_col]].dropna()
    if len(df_clean) < 2 or df_clean[temp_col].nunique() < 2:
        return None

    corr = df_clean[temp_col].corr(df_clean[emission_col])
    if pd.isna(corr):
        return None

    row_terpanas = df_clean.loc[df_clean[temp_col].idxmax()]
    row_terdingin = df_clean.loc[df_clean[temp_col].idxmin()]

    if lang == "id":
        if corr >= 0.5:
            arah = "ikut naik cukup jelas"
        elif corr >= 0.2:
            arah = "sedikit naik, tapi hubungannya nggak terlalu kuat"
        elif corr <= -0.5:
            arah = "justru turun"
        elif corr <= -0.2:
            arah = "sedikit turun, meski hubungannya lemah"
        else:
            arah = "nggak menunjukkan pola yang jelas"
        return (
            f"Kalau dilihat dari datanya, ketika suhu naik, emisi karbon di sini {arah} "
            f"(korelasinya {corr:.2f}). Contohnya, pas suhu paling tinggi ({row_terpanas[temp_col]:.1f}°C), "
            f"emisinya sekitar {row_terpanas[emission_col]:.2f} MT, sementara pas suhu paling rendah "
            f"({row_terdingin[temp_col]:.1f}°C), emisinya sekitar {row_terdingin[emission_col]:.2f} MT."
        )
    else:
        if corr >= 0.5:
            arah = "clearly tends to rise as well"
        elif corr >= 0.2:
            arah = "shows a slight upward tendency, though the link isn't very strong"
        elif corr <= -0.5:
            arah = "actually tends to fall"
        elif corr <= -0.2:
            arah = "shows a slight downward tendency, though weak"
        else:
            arah = "doesn't show a clear pattern"
        return (
            f"Looking at the data, carbon emissions here {arah} as temperature rises "
            f"(correlation of {corr:.2f}). For example, at the highest temperature "
            f"({row_terpanas[temp_col]:.1f}°C), emissions were around {row_terpanas[emission_col]:.2f} MT, "
            f"while at the lowest temperature ({row_terdingin[temp_col]:.1f}°C), emissions were around "
            f"{row_terdingin[emission_col]:.2f} MT."
        )


def build_comparison_insight(df_result: pd.DataFrame, lang: str = "id"):
    """
    Menghasilkan narasi perbandingan antar-wilayah SECARA OTOMATIS berdasarkan
    STRUKTUR data hasil query (bukan cuma kata kunci pertanyaan) -- jadi ini juga
    otomatis menangani kasus "selisih data aktual vs prediksi" yang sama-sama
    menghasilkan kolom region dengan beberapa nilai berbeda.
    Mengembalikan None kalau bukan kasus perbandingan.
    """
    if "region" not in df_result.columns or df_result["region"].nunique() < 2:
        return None

    numeric_cols = df_result.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None

    grouped = df_result.groupby("region")[numeric_cols].mean()
    lines = []

    for col in numeric_cols:
        sorted_regions = grouped[col].sort_values(ascending=False)
        top_region, top_val = sorted_regions.index[0], sorted_regions.iloc[0]
        bottom_region, bottom_val = sorted_regions.index[-1], sorted_regions.iloc[-1]
        selisih = top_val - bottom_val
        metric_label = humanize_metric(col, lang)

        if lang == "id":
            lines.append(
                f"{metric_label.capitalize()} paling tinggi ada di {top_region} ({top_val:.2f}), "
                f"paling rendah di {bottom_region} ({bottom_val:.2f}) — selisihnya {selisih:.2f}."
            )
        else:
            lines.append(
                f"The highest {metric_label} is in {top_region} ({top_val:.2f}), lowest in {bottom_region} "
                f"({bottom_val:.2f}) — a gap of {selisih:.2f}."
            )

    intro = "Bandingkan hasilnya:" if lang == "id" else "Here's the comparison:"
    return intro + "\n" + "\n".join(f"- {l}" for l in lines)


# 📌 Kata kunci yang menandakan pertanyaan bersifat REKOMENDASI/SARAN
# (bukan pertanyaan data), misalnya "apakah ada solusi untuk mengurangi emisi?"
# Pertanyaan seperti ini TIDAK PUNYA representasi SQL yang valid -> jangan dipaksa
# lewat Text-to-SQL (itu penyebab error "syntax error at or near 'Question'" sebelumnya).
ADVISORY_KEYWORDS = [
    "solusi", "cara mengurangi", "bagaimana cara", "gimana cara", "rekomendasi", "saran",
    "tips", "how to reduce", "suggest", "recommend", "advice", "mengurangi emisi",
    "reduce emission", "langkah apa", "apa yang bisa dilakukan"
]


def is_advisory_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in ADVISORY_KEYWORDS)


def get_latest_available_date():
    """Cari tanggal terbaru yang benar-benar ada di raw_data (dipakai untuk fallback
    ketika query CURRENT_DATE tidak menemukan data hari ini)."""
    try:
        row = db_conn.execute(
            "SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE generated_date IS NOT NULL"
        ).fetchone()
        return row[0] if row and row[0] is not None else None
    except Exception:
        return None


def build_anomaly_found_insight(df_result: pd.DataFrame, lang: str = "id"):
    """
    Merangkai jawaban natural saat query anomali (is_anomaly = 1) MENEMUKAN hasil --
    sebelumnya kasus ini jatuh ke jalur dump tabel mentah karena hanya kasus "kosong"
    yang ditangani khusus. Sekarang setiap baris anomali dijelaskan dalam kalimat,
    bukan tabel data mentah.
    """
    date_col = find_column(df_result, "date")
    temp_col = find_column(df_result, "temp")
    emission_col = find_column(df_result, "emission")
    region_col = "region" if "region" in df_result.columns else None
    n = len(df_result)

    lines = []
    for _, row in df_result.iterrows():
        where_bits = []
        if region_col:
            where_bits.append(f"di {row[region_col]}" if lang == "id" else f"in {row[region_col]}")
        if date_col:
            where_bits.append(f"pada {row[date_col]}" if lang == "id" else f"on {row[date_col]}")
        detail_bits = []
        if temp_col and pd.notna(row[temp_col]):
            detail_bits.append(f"suhu {row[temp_col]:.1f}°C" if lang == "id" else f"temperature {row[temp_col]:.1f}°C")
        if emission_col and pd.notna(row[emission_col]):
            detail_bits.append(f"emisi {row[emission_col]:.2f} MT" if lang == "id" else f"emissions of {row[emission_col]:.2f} MT")
        loc = " ".join(where_bits)
        detail = ", ".join(detail_bits)
        lines.append(f"{loc} — {detail}" if loc else detail)

    if lang == "id":
        intro = f"Ya, ada {n} anomali yang kedeteksi:" if n > 1 else "Ya, ada satu anomali yang kedeteksi:"
    else:
        intro = f"Yes, {n} anomalies were detected:" if n > 1 else "Yes, one anomaly was detected:"

    return intro + "\n" + "\n".join(f"- {l}" for l in lines)


def build_advisory_insight(user_q: str, lang: str = "id"):
    """
    Menjawab pertanyaan rekomendasi/saran TANPA lewat SQL generator (SLM kecil sering
    gagal membuat SQL valid untuk pertanyaan non-data seperti ini). Konteks data nyata
    diambil lewat query TETAP (bukan dari SLM) supaya tidak pernah gagal parse, lalu
    digabung dengan daftar tips praktis dalam satu bahasa sesuai pertanyaan.
    """
    try:
        df_ctx = db_conn.execute("""
            SELECT region,
                   AVG(carbon_emission_forecast_mt) AS avg_emission,
                   AVG(live_temperature) AS avg_temp
            FROM raw_data
            WHERE region IN ('Texas', 'Jakarta')
            GROUP BY region
        """).df()
    except Exception:
        df_ctx = pd.DataFrame()

    if lang == "id":
        if not df_ctx.empty:
            context = "; ".join(
                f"{r['region']} rata-rata emisinya {r['avg_emission']:.2f} MT (suhu rata-rata {r['avg_temp']:.1f}°C)"
                for _, r in df_ctx.iterrows()
            )
        else:
            context = "datanya belum tersedia"

        tips = [
            "Matikan lampu dan alat elektronik kalau lagi nggak dipakai.",
            "Coba pakai transportasi umum, sepeda, atau jalan kaki untuk jarak dekat.",
            "Kurangi konsumsi daging merah, perbanyak makanan nabati.",
            "Atur suhu AC secukupnya (sekitar 24-25°C), jangan kelewat dingin.",
            "Pilih peralatan elektronik hemat energi.",
            "Pas suhu lagi tinggi seperti di data di atas, konsumsi energi buat AC & kendaraan "
            "biasanya naik — jadi bisa dikurangi dulu pemakaiannya di hari-hari itu.",
        ]
        intro = f"Berdasarkan data yang ada ({context}), ini beberapa hal simpel yang bisa dicoba sehari-hari buat menekan emisi karbon:"
    else:
        if not df_ctx.empty:
            context = "; ".join(
                f"{r['region']} averages {r['avg_emission']:.2f} MT (avg temperature {r['avg_temp']:.1f}°C)"
                for _, r in df_ctx.iterrows()
            )
        else:
            context = "data isn't available yet"

        tips = [
            "Turn off lights and electronics when you're not using them.",
            "Try public transport, cycling, or walking for short trips.",
            "Cut back on red meat, eat more plant-based food.",
            "Keep your AC around 24-25°C instead of running it too cold.",
            "Go for energy-efficient appliances where you can.",
            "On hot days like the ones in the data above, energy use for cooling and transport "
            "tends to spike — worth cutting back a bit on those days.",
        ]
        intro = f"Based on the data ({context}), here are a few simple everyday things that can help lower carbon emissions:"

    return intro + "\n" + "\n".join(f"- {t}" for t in tips)


# ⚠️ PERBAIKAN PENTING:
# - Versi lama men-define endpoint /analyze DUA KALI secara nested dengan indentasi
#   salah -> ini yang menyebabkan IndentationError/SyntaxError sehingga server gagal jalan.
# - Versi lama juga memaksa SETIAP pertanyaan untuk selalu difilter ke
#   region ILIKE '%Jakarta%' dan generated_date ILIKE '%2021%06%29%' di system prompt.
#   Ini sisa data testing yang membuat AI tidak pernah benar-benar menjawab sesuai
#   pertanyaan user (mis. soal Texas atau tanggal lain tidak akan pernah terjawab benar).
#   Dua rule itu sudah dihapus di bawah.
@app.post("/analyze")
async def analyze_data(request: QueryRequest):
    # 📊 Pencatat waktu untuk benchmark fp16 vs 8-bit quantization -- latency_seconds
    # akan muncul di setiap response JSON, tanpa perlu instrumentasi terpisah.
    start_time = time.time()
    user_q = request.question
    lang = detect_language(user_q)

    if db_conn is None:
        msg = ("Data belum berhasil dimuat ke DuckDB, coba restart runtime Colab." if lang == "id"
               else "Data wasn't loaded into DuckDB, try restarting the Colab runtime.")
        return {"status": "error", "generated_sql": "", "raw_data": "No Data", "final_insight": msg,
                "latency_seconds": round(time.time() - start_time, 2)}

    # 🧠 Pertanyaan REKOMENDASI/SARAN (mis. "apakah ada solusi untuk mengurangi emisi?")
    # dijawab langsung di sini, TANPA lewat Text-to-SQL sama sekali -- pertanyaan seperti
    # ini memang tidak punya representasi SQL yang valid, jadi memaksanya lewat SLM SQL
    # generator hanya akan menghasilkan error parse seperti yang terjadi sebelumnya.
    if is_advisory_question(user_q):
        return {
            "status": "success",
            "generated_sql": "(tidak diperlukan - dijawab langsung dari data agregat)",
            "raw_data": "N/A",
            "final_insight": build_advisory_insight(user_q, lang),
            "latency_seconds": round(time.time() - start_time, 2)
        }

    # 📌 System prompt diberi CONTOH KONKRET (few-shot) mengikuti format data asli
    # (region persis "Texas"/"Jakarta"/"Texas (Prediction)"/"Jakarta (Prediction)",
    # dan generated_date persis "YYYY-MM-DD"). Model SLM sekecil Phi-3-mini jauh lebih
    # akurat menyusun SQL kalau diberi contoh langsung dibanding hanya deskripsi kolom.
    sql_messages = [
        {
            "role": "system",
            "content": """You are a strict DuckDB SQL generator. Output ONLY one valid SQL query, nothing else.

Table 'raw_data' columns:
- generated_date VARCHAR 'YYYY-MM-DD' | generated_at VARCHAR 'YYYY-MM-DD HH:MM:SS'
- region VARCHAR, exact values: 'Texas', 'Jakarta', 'Texas (Prediction)', 'Jakarta (Prediction)'
- live_temperature DOUBLE | base_emission_mt DOUBLE | carbon_emission_forecast_mt DOUBLE
- prophet_upper DOUBLE, prophet_lower DOUBLE: batas atas/bawah forecast 30 hari (HANYA terisi untuk region '... (Prediction)', NULL untuk data aktual)
- is_anomaly INTEGER: 0 = Normal, 1 = Anomaly, NULL = model belum cukup data untuk dilatih (bukan error)
- recommended_carbon_cap_mt DOUBLE: batas emisi yang direkomendasikan (formula: Normal -> forecast x0.95, Anomaly -> baseline x1.05)

RULES:
1. Always LOWER(region) = LOWER('<value>') for exact match, or LOWER(region) LIKE LOWER('<prefix>%') to catch both actual + prediction of a city.
2. Default emission metric = carbon_emission_forecast_mt (this is what the dashboard shows). Use base_emission_mt only if user explicitly says "baseline"/"sebelum penyesuaian".
3. Explicit date mentioned -> literal 'YYYY-MM-DD'. Relative date ("hari ini", "minggu ini", "N hari terakhir") -> DO NOT use CURRENT_DATE (server clock timezone may not match the data's actual date). Instead anchor to the latest ACTUAL date present in the table: (SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE region NOT LIKE '%Prediction%').
4. "Kapan/di mana nilai tertinggi/terendah" -> SELECT the identifier columns (generated_date, region) together with the metric, ORDER BY metric DESC/ASC, LIMIT 1. Never SELECT just MAX()/MIN() alone for these questions.
5. Use clean snake_case aliases (AS avg_temp, AS avg_emission) for every aggregated column.
6. No WHERE clause at all if the question mentions no region or date.
7. Wrap output in exactly one ```sql code block```, nothing before or after.
8. "Anomali/lonjakan tidak wajar" -> filter is_anomaly = 1. Never filter is_anomaly for prediction rows (they don't have anomaly labels).
9. "Batas aman/carbon cap/rekomendasi batas emisi" -> use recommended_carbon_cap_mt column (only exists for actual data rows, not predictions).
10. "Rentang/interval ketidakpastian forecast" -> include prophet_upper and prophet_lower alongside carbon_emission_forecast_mt, filtered to '... (Prediction)' region.

Examples:

Q: Berapa rata-rata emisi karbon di Jakarta pada 29 Juni?
```sql
SELECT AVG(carbon_emission_forecast_mt) AS avg_emission FROM raw_data WHERE LOWER(region) = LOWER('Jakarta') AND generated_date = '2026-06-29'
```

Q: Berapa rata-rata emisi baseline di Texas?
```sql
SELECT AVG(base_emission_mt) AS avg_baseline_emission FROM raw_data WHERE LOWER(region) = LOWER('Texas')
```

Q: Bandingkan emisi karbon di Jakarta dan Texas hari ini.
```sql
SELECT region, generated_date, carbon_emission_forecast_mt FROM raw_data WHERE LOWER(region) IN (LOWER('Jakarta'), LOWER('Texas')) AND CAST(generated_date AS DATE) = (SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE region NOT LIKE '%Prediction%')
```

Q: Kapan suhu tertinggi di Jakarta terjadi dan berapa nilainya?
```sql
SELECT generated_date, live_temperature FROM raw_data WHERE LOWER(region) = LOWER('Jakarta') ORDER BY live_temperature DESC LIMIT 1
```

Q: Tampilkan proyeksi emisi Jakarta untuk 3 hari ke depan
```sql
SELECT generated_date, carbon_emission_forecast_mt FROM raw_data WHERE LOWER(region) = LOWER('Jakarta (Prediction)') ORDER BY generated_date
```

Q: Berapa selisih emisi Jakarta saat ini dengan emisinya di masa prediksi?
```sql
SELECT region, AVG(carbon_emission_forecast_mt) AS avg_emission FROM raw_data WHERE LOWER(region) LIKE LOWER('Jakarta%') GROUP BY region
```

Q: Tampilkan data emisi Texas selama 7 hari terakhir.
```sql
SELECT generated_date, carbon_emission_forecast_mt FROM raw_data WHERE LOWER(region) = LOWER('Texas') AND CAST(generated_date AS DATE) >= (SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE region NOT LIKE '%Prediction%') - INTERVAL '7 days'
```

Q: Jika ada kenaikan suhu di Texas apakah emisi di Texas juga naik?
```sql
SELECT generated_date, AVG(live_temperature) AS avg_temp, AVG(carbon_emission_forecast_mt) AS avg_emission FROM raw_data WHERE LOWER(region) = LOWER('Texas') GROUP BY generated_date ORDER BY generated_date
```

Q: Berapa rata-rata suhu dari seluruh data yang ada?
```sql
SELECT AVG(live_temperature) AS avg_temp FROM raw_data
```

Q: Apakah ada anomali emisi di Jakarta minggu ini?
```sql
SELECT generated_date, live_temperature, carbon_emission_forecast_mt FROM raw_data WHERE LOWER(region) = LOWER('Jakarta') AND is_anomaly = 1 AND CAST(generated_date AS DATE) >= (SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE region NOT LIKE '%Prediction%') - INTERVAL '7 days'
```

Q: Berapa batas emisi (carbon cap) yang direkomendasikan untuk Texas hari ini?
```sql
SELECT generated_date, recommended_carbon_cap_mt FROM raw_data WHERE LOWER(region) = LOWER('Texas') AND CAST(generated_date AS DATE) = (SELECT MAX(CAST(generated_date AS DATE)) FROM raw_data WHERE region NOT LIKE '%Prediction%')
```

Q: Tampilkan proyeksi emisi Jakarta 30 hari ke depan beserta rentang ketidakpastiannya
```sql
SELECT generated_date, carbon_emission_forecast_mt, prophet_lower, prophet_upper FROM raw_data WHERE LOWER(region) = LOWER('Jakarta (Prediction)') ORDER BY generated_date
```"""
        },
        {"role": "user", "content": f"Question: {user_q}\nSQL:"}
    ]
    raw_sql = generate_slm_response(sql_messages, max_tokens=130)
    generated_sql = clean_sql_string(raw_sql)

    try:
        df_result = db_conn.execute(generated_sql).df()
        fallback_note = ""

        # 🔄 FALLBACK OTOMATIS: kalau query pakai CURRENT_DATE tapi hasilnya kosong,
        # kemungkinan besar pipeline BELUM dijalankan untuk hari ini sehingga data
        # "hari ini" memang belum ada -- bukan berarti SQL-nya salah. Coba ulang query
        # yang sama dengan tanggal TERBARU yang benar-benar ada di data lake.
        if df_result.empty and "CURRENT_DATE" in generated_sql.upper():
            latest_date = get_latest_available_date()
            if latest_date is not None:
                fallback_sql = re.sub(r"CURRENT_DATE", f"DATE '{latest_date}'", generated_sql, flags=re.IGNORECASE)
                try:
                    df_fallback = db_conn.execute(fallback_sql).df()
                    if not df_fallback.empty:
                        df_result = df_fallback
                        generated_sql = fallback_sql
                        fallback_note = (
                            f"(Data hari ini belum ada di data lake, mungkin pipeline belum jalan hari ini -- "
                            f"ini data terbaru yang tersedia, per {latest_date}.) "
                            if lang == "id" else
                            f"(Today's data isn't in the data lake yet, the pipeline may not have run today -- "
                            f"showing the most recent available data, from {latest_date}.) "
                        )
                except Exception:
                    pass

        # 🔍 DIAGNOSTIK KHUSUS UNTUK PERTANYAAN ANOMALI: kalau filter is_anomaly = 1
        # hasilnya kosong, itu bisa berarti dua hal yang beda maknanya: (a) model belum
        # selesai dilatih (is_anomaly masih NULL), atau (b) memang tidak ada anomali sama
        # sekali (kabar baik!). Cek status sebenarnya supaya jawabannya jelas.
        anomaly_diagnosed = False
        if df_result.empty and re.search(r"is_anomaly\s*=\s*1\b", generated_sql, re.IGNORECASE):
            where_match = re.search(r"WHERE\s+(.*)$", generated_sql, re.IGNORECASE)
            if where_match:
                where_no_anomaly = re.sub(
                    r"\s*AND\s+is_anomaly\s*=\s*1\b|is_anomaly\s*=\s*1\b\s*AND\s*",
                    "", where_match.group(1), flags=re.IGNORECASE
                ).strip()
                try:
                    diagnostic_sql = f"SELECT is_anomaly, COUNT(*) AS cnt FROM raw_data WHERE {where_no_anomaly} GROUP BY is_anomaly"
                    df_diag = db_conn.execute(diagnostic_sql).df()
                    if not df_diag.empty:
                        if df_diag["is_anomaly"].isna().all():
                            final_insight = (
                                "Model deteksi anomalinya belum selesai dilatih buat periode ini (butuh minimal "
                                "30 baris histori aktual dulu), jadi belum bisa dinilai anomali atau nggak. "
                                "Coba tanya lagi setelah pipeline jalan beberapa hari lagi ya."
                                if lang == "id" else
                                "The anomaly detection model hasn't finished training yet for this period (needs "
                                "at least 30 rows of actual history), so it can't be evaluated yet. Try again "
                                "after the pipeline has run for a few more days."
                            )
                        else:
                            n_normal = int(df_diag.loc[df_diag["is_anomaly"] == 0, "cnt"].sum()) if 0 in df_diag["is_anomaly"].values else 0
                            final_insight = (
                                f"Nggak ada anomali di periode ini — kabar baik, emisi terpantau normal "
                                f"({n_normal} data diperiksa)."
                                if lang == "id" else
                                f"No anomalies found for this period — good news, emissions look normal "
                                f"({n_normal} data points checked)."
                            )
                        anomaly_diagnosed = True
                except Exception:
                    pass

        if anomaly_diagnosed:
            pass
        elif df_result.empty:
            final_insight = (
                f"Query-nya berhasil jalan tapi nggak ada data yang cocok. Coba cek lagi tanggal atau "
                f"wilayah di pertanyaan Anda ya. (SQL: {generated_sql})"
                if lang == "id" else
                f"The query ran fine but returned no matching data. Try double-checking the date or "
                f"region in your question. (SQL: {generated_sql})"
            )
        elif len(df_result.columns) == 1 and pd.api.types.is_numeric_dtype(df_result.iloc[:, 0]):
            nilai = df_result.iloc[0, 0]
            if pd.isna(nilai):
                final_insight = (
                    f"Nggak ketemu data yang cocok buat pertanyaan ini. Kemungkinan tanggal atau wilayah "
                    f"yang dimaksud belum ada di data lake -- coba pertegas tanggalnya. (SQL: {generated_sql})"
                    if lang == "id" else
                    f"Couldn't find matching data for this question. The date or region might not exist "
                    f"in the data lake yet -- try being more specific about the date. (SQL: {generated_sql})"
                )
            else:
                metric_label = humanize_metric(df_result.columns[0], lang)
                final_insight = fallback_note + (
                    f"{metric_label.capitalize()} sekitar {nilai:.2f}."
                    if lang == "id" else
                    f"The {metric_label} is about {nilai:.2f}."
                )
        elif re.search(r"is_anomaly\s*=\s*1\b", generated_sql, re.IGNORECASE):
            # 🚨 Query anomali yang MENEMUKAN hasil (bukan kosong) -> jawab dengan
            # kalimat natural per baris, bukan dump tabel mentah.
            final_insight = fallback_note + build_anomaly_found_insight(df_result, lang)
        elif build_comparison_insight(df_result, lang) is not None:
            # 🆚 Kolom region punya lebih dari satu nilai -> ini pertanyaan perbandingan
            # antar-wilayah (atau aktual vs prediksi).
            final_insight = fallback_note + build_comparison_insight(df_result, lang)
        elif is_analytical_question(user_q) and build_correlation_insight(df_result, user_q, lang) is not None:
            # 🧠 Pertanyaan bersifat logika/sebab-akibat -> jawab dengan analisis korelasi.
            final_insight = fallback_note + build_correlation_insight(df_result, user_q, lang)
        else:
            # Jawaban umum non-analitis: batasi tampilan supaya tetap natural dibaca.
            if len(df_result) > 10:
                preview_str = df_result.head(10).to_string(index=False)
                intro = (
                    f"Ketemu {len(df_result)} baris data. Ini contoh 10 baris pertamanya:"
                    if lang == "id" else
                    f"Found {len(df_result)} rows. Here's a sample of the first 10:"
                )
                final_insight = fallback_note + f"{intro}\n{preview_str}"
            else:
                query_result_str = df_result.to_string(index=False)
                intro = "Ini hasilnya:" if lang == "id" else "Here's the result:"
                final_insight = fallback_note + f"{intro}\n{query_result_str}"

    except Exception as e:
        df_result = pd.DataFrame()
        final_insight = (
            f"Gagal memproses query-nya: {str(e)} (SQL: {generated_sql})"
            if lang == "id" else
            f"Failed to process the query: {str(e)} (SQL: {generated_sql})"
        )

    elapsed = round(time.time() - start_time, 2)
    print(f"⏱️ Latency /analyze: {elapsed}s (mode: {'8-bit' if FORCE_8BIT_QUANTIZATION else 'fp16'})")

    return {
        "status": "success",
        "generated_sql": generated_sql,
        "raw_data": df_result.to_string(index=False) if not df_result.empty else "No Data",
        "final_insight": final_insight,
        "latency_seconds": elapsed
    }


# ==========================================
# JALANKAN TUNNEL (CLOUDFLARE TUNNEL -- GRATIS, TANPA AKUN/TOKEN)
# ==========================================
# ⚠️ MIGRASI DARI NGROK KE CLOUDFLARE TUNNEL:
# ngrok free tier terbukti (lewat pengujian berulang -- browser & Traffic Inspector
# sama-sama gagal, bahkan dengan domain custom & header anti-abuse) memutus koneksi
# di tengah TLS handshake untuk traffic dari datacenter-ke-datacenter (HF Space ->
# Colab), menghasilkan SSLEOFError yang konsisten. Cloudflare Tunnel historisnya jauh
# lebih toleran terhadap pola traffic seperti ini, dan GRATIS TANPA PERLU DAFTAR AKUN
# ATAU TOKEN SAMA SEKALI untuk quick tunnel (`trycloudflare.com`).
#
# ⚠️ TRADE-OFF: quick tunnel gratis ini URL-nya ACAK tiap sesi (mis.
# https://xxxx-xxxx-xxxx.trycloudflare.com) -- BEDA dengan ngrok static domain yang
# permanen. Kalau nanti Cloudflare Tunnel ini terbukti stabil, domain permanen bisa
# didapat gratis juga tapi perlu domain sendiri terdaftar di Cloudflare (langkah
# terpisah, tidak dibahas di sini karena prioritas sekarang adalah BERFUNGSI dulu).

print("⬇️  Mengunduh cloudflared (sekali per sesi Colab)...")
get_ipython().system_raw("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared")
time.sleep(3)  # beri waktu proses download/chmod selesai sebelum dipakai

if os.path.exists("cf_tunnel.log"):
    os.remove("cf_tunnel.log")

get_ipython().system_raw("./cloudflared tunnel --url http://localhost:8000 > cf_tunnel.log 2>&1 &")

# 🔎 POLLING: cloudflared butuh beberapa detik untuk konek ke edge Cloudflare dan
# dapat subdomain trycloudflare.com -- cek log tiap 1 detik sampai URL muncul,
# dengan timeout supaya tidak nunggu selamanya kalau memang gagal.
tunnel_url = None
for _ in range(30):
    if os.path.exists("cf_tunnel.log"):
        with open("cf_tunnel.log", "r") as f:
            content = f.read()
        match = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", content)
        if match:
            tunnel_url = match.group(0)
            break
    time.sleep(1)

print("\n=======================================================")
if tunnel_url:
    print("🌍 LINK TUNNEL (TEMPEL KE STREAMLIT):")
    print(tunnel_url)
    print("=======================================================\n")
    print("⚠️ CATATAN: URL ini ACAK dan akan BERUBAH setiap kali sel ini dijalankan")
    print("   ulang -- update manual ke Streamlit tiap kali restart Colab.")
    print("ℹ️ Tes cepat: buka " + tunnel_url + "/health di browser untuk cek server hidup")
    print("   tanpa memicu inferensi model.")
else:
    print("❌ GAGAL mendapatkan URL tunnel Cloudflare setelah 30 detik. Isi cf_tunnel.log:")
    if os.path.exists("cf_tunnel.log"):
        with open("cf_tunnel.log", "r") as f:
            print(f.read().strip() or "(file kosong)")
    else:
        print("(cf_tunnel.log belum terbuat -- kemungkinan proses cloudflared gagal start)")
    print("=======================================================")

config = uvicorn.Config(app, host="127.0.0.1", port=8000, loop="asyncio")
server = uvicorn.Server(config)
await server.serve()
