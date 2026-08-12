import streamlit as st
import streamlit.components.v1 as components
import os
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from huggingface_hub import hf_hub_download

# 1. Konfigurasi halaman Streamlit (Wajib di baris paling pertama)
st.set_page_config(
    page_title="Climate-Driven Carbon Emission Pipeline",
    page_icon="🌱",
    layout="wide"
)

# ⚙️ KONFIGURASI DATA LAKE (harus sama persis dengan pipeline-prefect.py)
HF_REPO_ID = "sigit48/carbon-emission-datalake"
HF_REMOTE_PATH = "data/carbon_emission_master.parquet"

# ⚙️ KONFIGURASI GITHUB ACTIONS (isi via Space Secrets)
# ⚠️ ARSITEKTUR BARU: pipeline TIDAK LAGI dijalankan langsung di dalam proses
# Space (dulu lewat subprocess.run(["python3","pipeline-prefect.py"])) --
# sekarang cuma MEMICU workflow GitHub Actions yang menjalankannya di runner
# terpisah. Alasan migrasi:
#   1. HF Space free tier sekarang membatasi/menghilangkan CPU Basic gratis
#      untuk Docker/Gradio (kebijakan baru per pertengahan 2026), sementara
#      GitHub Actions tetap gratis tanpa batas untuk repo publik.
#   2. Beban CPU berat (Prophet+dbt+scikit-learn) yang jalan di dalam proses
#      Space kemungkinan turut berkontribusi ke masalah stabilitas Space
#      sebelumnya -- memindahkannya keluar mengurangi risiko itu.
#   3. Pipeline jadi bisa terjadwal otomatis (lihat .github/workflows/run_pipeline.yml)
#      tanpa perlu Space-nya "hidup" sama sekali.
GITHUB_REPO = os.environ.get("GITHUB_REPO", "")            # format: "username/nama-repo"
GITHUB_PAT = os.environ.get("GITHUB_PAT", "")               # personal access token, scope 'repo' atau 'actions:write'
GITHUB_WORKFLOW_FILE = os.environ.get("GITHUB_WORKFLOW_FILE", "run_pipeline.yml")
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main")


def create_api_session():
    """
    Session dibuat lewat factory function supaya bisa dengan mudah DIBUAT ULANG
    (bukan cuma dipakai ulang) kalau ada kegagalan koneksi -- retry otomatis di
    level adapter untuk error transient, plus retry-with-fresh-session di kode
    pemanggil untuk kasus koneksi yang sudah rusak di level TLS.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[502, 503, 504],
        allowed_methods=["POST", "GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def trigger_github_pipeline():
    """
    Memicu workflow GitHub Actions lewat REST API (workflow_dispatch), bukan
    menjalankan pipeline secara lokal di proses Space. Mengembalikan (sukses: bool, pesan: str).
    """
    if not GITHUB_REPO or not GITHUB_PAT:
        return False, (
            "GITHUB_REPO dan/atau GITHUB_PAT belum diisi di Space Secrets. "
            "Tanpa ini, pipeline tidak bisa dipicu dari sini -- tapi tetap jalan "
            "otomatis sesuai jadwal cron di GitHub Actions."
        )
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW_FILE}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_PAT}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    try:
        resp = requests.post(url, headers=headers, json={"ref": GITHUB_BRANCH}, timeout=15)
        if resp.status_code == 204:
            return True, "Pipeline berhasil dipicu di GitHub Actions! Cek progres di tab Actions repo Anda."
        else:
            return False, f"Gagal memicu workflow (HTTP {resp.status_code}): {resp.text[:300]}"
    except Exception as e:
        return False, f"Gagal menghubungi GitHub API: {str(e)}"


@st.cache_data(ttl=120)
def load_latest_data_from_hf():
    """
    Baca data langsung dari parquet di Hugging Face -- BUKAN dari file DuckDB
    lokal (your_project.db). Karena pipeline sekarang jalan di GitHub Actions
    runner yang sifatnya EPHEMERAL, file lokal itu tidak pernah tersimpan di
    Space. Data lake HF adalah satu-satunya sumber kebenaran yang persisten,
    jadi Space membaca dari situ langsung. Di-cache 120 detik supaya tidak
    berulang kali download tiap re-run Streamlit.
    """
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_REMOTE_PATH, repo_type="dataset")
        df = pd.read_parquet(path)
        return df, None
    except Exception as e:
        return None, str(e)


# ⚡ Inisialisasi HTTP Session di tingkat global
if "api_session" not in st.session_state:
    st.session_state.api_session = create_api_session()

# Inisialisasi penyimpanan state lainnya agar data tidak hilang saat re-run
if "last_ai_result" not in st.session_state:
    st.session_state.last_ai_result = None
if "last_trigger_status" not in st.session_state:
    st.session_state.last_trigger_status = None
if "last_trigger_message" not in st.session_state:
    st.session_state.last_trigger_message = None

# Title & Deskripsi Portofolio
st.title("🌱 Climate-Driven Carbon Emission Automation Pipeline")
st.markdown("""**Welcome to my Modern Data Stack Portfolio!** Aplikasi ini mendemonstrasikan pipeline data otomatis (*End-to-End*) menggunakan **Prefect**, **DuckDB**, **dbt (Data Build Tool)**, **GitHub Actions**, **Google Sheets API**, dan **Looker Studio**.""")

# Pembagian Layout menjadi 2 Kolom
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("⚙️ Pipeline Controller")
    st.write(
        "Pipeline berjalan otomatis terjadwal via **GitHub Actions** (bukan di dalam Space ini). "
        "Klik tombol di bawah untuk memicu run manual, atau biarkan berjalan sesuai jadwal."
    )

    if st.button("🚀 Trigger Pipeline (via GitHub Actions)", type="primary", key="trigger_pipeline_btn"):
        with st.spinner("Memicu workflow di GitHub Actions..."):
            success, message = trigger_github_pipeline()
            st.session_state.last_trigger_status = "success" if success else "error"
            st.session_state.last_trigger_message = message

    if st.session_state.last_trigger_status == "success":
        st.success(f"✓ {st.session_state.last_trigger_message}")
        if GITHUB_REPO:
            st.link_button("Lihat progres di GitHub Actions", f"https://github.com/{GITHUB_REPO}/actions")
    elif st.session_state.last_trigger_status == "error":
        st.warning(st.session_state.last_trigger_message)

    st.markdown("---")
    st.subheader("📊 Database Preview (Data Lake HF)")

    df_preview, load_error = load_latest_data_from_hf()
    if df_preview is not None:
        df_sorted = df_preview.sort_values("generated_at", ascending=False).head(10)
        st.dataframe(df_sorted, use_container_width=True)
        st.caption(f"Total baris di data lake: {len(df_preview)} | Preview di-cache 2 menit")
    else:
        st.info("💡 Data lake belum bisa diakses. Kemungkinan pipeline belum pernah jalan sama sekali, "
                f"atau ada masalah koneksi. Detail: {load_error}")

    # ==========================================
    # 🤖 ⚡ SEKSI BARU: AUTONOMOUS SLM ESG ANALYST (DYNAMIC ROUTING)
    # ==========================================
    st.markdown("---")
    st.subheader("🤖 Autonomous AI ESG Analyst (Phi-3 SLM)")
    st.write("Ajukan pertanyaan analisis data emisi karbon, AI akan otomatis membuat kueri SQL dan menganalisisnya secara bilingual.")

    # 🔗 INPUT URL DINAMIS (Cloudflare Tunnel, URL acak tiap sesi Colab)
    SLM_API_BASE = st.text_input(
        "🔗 Masukkan URL Tunnel dari Colab (Cloudflare Tunnel):",
        value="",
        placeholder="Contoh: https://random-word-combo-1234.trycloudflare.com",
        key="slm_api_tunnel_url"
    )

    if SLM_API_BASE.endswith("/analyze"):
        SLM_API_URL = SLM_API_BASE
    else:
        SLM_API_URL = f"{SLM_API_BASE.rstrip('/')}/analyze"

    user_query = st.text_input(
        "Masukkan pertanyaan kamu:",
        placeholder="Contoh: Berapa rata-rata emisi karbon di Jakarta?",
        key="slm_analyst_input"
    )

    if st.button("🧠 Tanya AI Analyst", type="secondary", key="ask_slm_btn"):
        if not user_query.strip():
            st.warning("Silakan masukkan pertanyaan terlebih dahulu!")
        elif not SLM_API_BASE.strip() or not SLM_API_BASE.strip().startswith("https://"):
            st.warning("Silakan masukkan URL Tunnel dari Colab yang valid (harus diawali https://).")
        else:
            with st.spinner("AI sedang menerjemahkan pertanyaan ke SQL dan menganalisis Data Lake..."):
                payload = {"question": user_query}
                attempts_left = 2
                last_error = None
                success = False

                while attempts_left > 0 and not success:
                    attempts_left -= 1
                    try:
                        response = st.session_state.api_session.post(
                            SLM_API_URL,
                            json=payload,
                            timeout=55
                        )
                        if response.status_code == 200:
                            st.session_state.last_ai_result = response.json()
                            st.success("🎯 Analisis Selesai!")
                            success = True
                        else:
                            st.error(f"Gagal terhubung ke model. Kode Status Server: {response.status_code}")
                            success = True

                    except requests.exceptions.Timeout:
                        st.warning("⚠️ Model di GPU Colab memerlukan waktu ekstra untuk menyusun laporan. Silakan klik kembali tombol **🧠 Tanya AI Analyst** untuk menarik data hasil akhir.")
                        success = True

                    except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as conn_err:
                        last_error = conn_err
                        if attempts_left > 0:
                            st.session_state.api_session = create_api_session()
                        continue

                    except Exception as api_err:
                        last_error = api_err
                        break

                if not success and last_error is not None:
                    st.error(f"Terjadi kesalahan koneksi API (setelah dicoba ulang): {str(last_error)}")
                    st.info("Pastikan Google Colab kamu sudah menyala dan URL Tunnel Cloudflare yang ditempel "
                            "adalah yang TERBARU (URL ini acak dan berubah tiap restart Colab).")

    if st.session_state.last_ai_result is not None:
        res_data = st.session_state.last_ai_result
        st.markdown(f"### 📋 Executive Insight:\n{res_data['final_insight']}")

        with st.expander("🔍 Lihat Proses Kerja AI (Text-to-SQL & Raw Data)", expanded=False):
            st.markdown("**Generated SQL Query (DuckDB):**")
            st.code(res_data['generated_sql'], language='sql')
            st.markdown("**Raw Data Output dari Data Lake Parquet:**")
            st.code(res_data['raw_data'])

# 💡 BAGIAN UTAMA LAYOUT KANAN
with col2:
    st.subheader("📊 Looker Studio Live Dashboard")

    LOOKER_STUDIO_EMBED_URL = "https://datastudio.google.com/embed/reporting/aaabe554-3853-4601-992a-7cbd580cbf70/page/bv81F"

    components.iframe(LOOKER_STUDIO_EMBED_URL, height=750)

    st.warning("⚠️ **Catatan Visualisasi:** Jika dashboard Looker Studio di bawah tidak muncul atau meminta izin cookie, Anda dapat mengaksesnya langsung melalui tautan resmi berikut.")
    st.link_button("Buka Dashboard di Looker Studio", "https://lookerstudio.google.com/reporting/aaabe554-3853-4601-992a-7cbd580cbf70/page/bv81F")
