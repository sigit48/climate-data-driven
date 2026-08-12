-- Test kustom (singular test): carbon_emission_forecast_mt tidak boleh negatif.
-- Konvensi dbt: test GAGAL kalau query ini mengembalikan >0 baris, LOLOS kalau 0 baris.
-- Tidak butuh package tambahan (dbt_utils/dbt_expectations) -- murni SQL biasa.

SELECT *
FROM {{ ref('stg_carbon_emissions') }}
WHERE carbon_emission_forecast_mt < 0
