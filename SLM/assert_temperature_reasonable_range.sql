-- Test kustom (singular test): suhu di luar rentang wajar (-10°C s/d 55°C) untuk
-- Texas & Jakarta kemungkinan besar indikasi bug/data error, bukan cuaca ekstrem asli.
-- Hanya berlaku untuk baris AKTUAL (bukan prediksi, karena Prophet tidak menghasilkan
-- nilai live_temperature).
-- Konvensi dbt: test GAGAL kalau query ini mengembalikan >0 baris, LOLOS kalau 0 baris.

SELECT *
FROM {{ ref('stg_carbon_emissions') }}
WHERE region NOT LIKE '%Prediction%'
  AND (live_temperature < -10 OR live_temperature > 55)
