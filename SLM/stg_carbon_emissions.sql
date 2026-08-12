WITH raw_data AS (
    SELECT * FROM {{ source('raw', 'carbon_emissions') }}
)
SELECT
    CAST(generated_at AS DATE) AS event_date,
    CAST(generated_at AS TIMESTAMP) AS created_at,
    region,
    live_temperature,
    base_emission_mt,
    carbon_emission_forecast_mt,
    prophet_upper,
    prophet_lower,
    is_anomaly,
    recommended_carbon_cap_mt
FROM raw_data
