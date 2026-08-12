# Checklist Deploy: GitHub + Streamlit Community Cloud

## 1. Push semua file ini ke repo GitHub (PUBLIK, untuk tier gratis)
Struktur folder di paket ini SUDAH BENAR, tinggal push apa adanya:
```bash
git init
git add .
git commit -m "Initial commit: carbon emission pipeline"
git branch -M main
git remote add origin https://github.com/<username>/<nama-repo>.git
git push -u origin main
```

⚠️ JANGAN commit file `.streamlit/secrets.toml` kalau Anda buat untuk testing
lokal -- itu HARUS masuk `.gitignore`. Secrets diisi lewat UI Streamlit Cloud,
BUKAN lewat file di repo.

## 2. Setup GitHub Actions Secrets (untuk pipeline-prefect.py)
Di repo GitHub -> Settings -> Secrets and variables -> Actions -> New repository secret:
- `GOOGLE_CREDENTIALS` -- isi JSON service account Google Sheets Anda (satu baris/compact JSON)
- `HF_TOKEN` -- token Hugging Face (kalau dataset private; boleh dilewati kalau publik)
- `NOTIFY_WEBHOOK_URL` -- opsional, webhook Slack/Discord untuk notifikasi kegagalan

Setelah ini, workflow `.github/workflows/run_pipeline.yml` akan otomatis jalan
terjadwal (setiap hari 03:00 UTC), atau bisa dipicu manual dari tab "Actions"
di GitHub (tombol "Run workflow").

## 3. Test manual dulu pipeline-nya di GitHub Actions
SEBELUM connect ke Streamlit, pastikan pipeline jalan dengan benar dulu:
- Buka tab "Actions" di repo -> pilih workflow "Carbon Emission Pipeline"
  -> klik "Run workflow" -> pilih branch main -> Run.
- Tunggu selesai (~5-10 menit karena instalasi Prophet cukup lama), cek log-nya
  hijau semua (Completed).

## 4. Buat Personal Access Token (PAT) untuk trigger dari Streamlit
Ini dipakai supaya tombol "Trigger Pipeline" di Streamlit bisa memicu GitHub
Actions dari luar:
- GitHub -> Settings (akun, bukan repo) -> Developer settings ->
  Personal access tokens -> Fine-grained tokens -> Generate new token
- Repository access: pilih repo ini saja
- Permissions: "Actions" -> Read and write
- Simpan token-nya (cuma muncul sekali!)

## 5. Deploy ke Streamlit Community Cloud
- Buka https://share.streamlit.io -> Sign in pakai akun GitHub
- "Create app" -> pilih repo, branch `main`, main file path `app.py`
- Klik "Advanced settings" SEBELUM deploy, isi bagian "Secrets" (format TOML,
  root-level, BUKAN di bawah [section] -- supaya otomatis kebaca lewat
  os.environ.get() di app.py):
```toml
GITHUB_REPO = "username/nama-repo"
GITHUB_PAT = "github_pat_xxxxxxxxxxxx"
GITHUB_WORKFLOW_FILE = "run_pipeline.yml"
GITHUB_BRANCH = "main"
```
- Klik "Deploy!"

## 6. Setelah live, jalankan Colab SLM server
- Buka `colab_slm_server.py` di Google Colab (upload manual atau lewat GitHub
  raw link), jalankan semua sel seperti biasa.
- Copy URL `https://xxxx-xxxx.trycloudflare.com` yang muncul di output.
- Tempel ke kolom "URL Tunnel dari Colab" di aplikasi Streamlit yang sudah live.

## 7. Testing akhir -- JANGAN AGRESIF
- Klik "Trigger Pipeline" SEKALI, cek tab GitHub Actions untuk lihat progresnya.
- Coba "Tanya AI Analyst" dengan jeda antar-percobaan (bukan spam klik cepat).

## Yang BEDA dari setup HF Space lama
- Pipeline TIDAK LAGI jalan di dalam proses Space/App -- sekarang di GitHub
  Actions runner terpisah (gratis tanpa batas untuk repo publik).
- "Database Preview" di Streamlit sekarang baca LANGSUNG dari parquet HF
  (bukan file DuckDB lokal), karena file lokal itu bersifat sementara di
  runner GitHub Actions yang ephemeral.
- Tidak butuh Docker sama sekali di sisi hosting dashboard.
