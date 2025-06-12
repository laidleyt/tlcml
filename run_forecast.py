import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime
import os
from dateutil.relativedelta import relativedelta

INPUT_PARQUET = "data/forecast_input.parquet"
OUTPUT_PARQUET = "data/forecast_output.parquet"
FITTED_PARQUET = "data/forecast_fitted.parquet"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_next_forecast_window(df):
    df["month"] = df["ds"].dt.to_period("M")
    month_counts = df["month"].value_counts()
    full_months = month_counts[month_counts >= 28].sort_index()
    if full_months.empty:
        log("[ERROR] No full month of actuals available.")
        return None, None

    latest_month = full_months.index[-1]
    forecast_start = (latest_month + 1).to_timestamp()
    forecast_end = (forecast_start + relativedelta(months=1)) - pd.Timedelta(days=1)
    return forecast_start, forecast_end

def forecast_and_save():
    if not os.path.exists(INPUT_PARQUET):
        log("[ERROR] Input file not found.")
        return

    df = pd.read_parquet(INPUT_PARQUET).rename(columns={"trip_date": "ds", "total_rides": "y"})
    df["ds"] = pd.to_datetime(df["ds"]).dt.normalize()
    df = df.dropna(subset=["ds", "y"]).drop_duplicates("ds").sort_values("ds")
    df = df[df["ds"] >= "2020-03-01"]  # Enforce start date for seasonality stability

    log(f"[DEBUG] Training from {df['ds'].min().date()} to {df['ds'].max().date()} — {len(df)} rows")

    forecast_start, forecast_end = get_next_forecast_window(df)
    if forecast_start is None:
        return

    # 🔒 Skip if forecast already covers this month
    if os.path.exists(OUTPUT_PARQUET):
        try:
            df_forecast = pd.read_parquet(OUTPUT_PARQUET)
            df_forecast["ds"] = pd.to_datetime(df_forecast["ds"])
            latest_forecast_date = df_forecast["ds"].max()
            if latest_forecast_date >= forecast_end:
                log(f"[SKIP] Forecast already up to date through {latest_forecast_date.date()}")
                return
        except Exception as e:
            log(f"[WARN] Could not read existing forecast file: {e} — proceeding with forecast.")

    # ─── Forecast model ───
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.1
    )
    model.add_seasonality(name="daily", period=1, fourier_order=5)
    model.fit(df[["ds", "y"]])

    # ─── Forecast next month ───
    future_df = pd.DataFrame({"ds": pd.date_range(start=forecast_start, end=forecast_end, freq="D")})
    forecast = model.predict(future_df)

    forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_df[["yhat", "yhat_lower", "yhat_upper"]] = forecast_df[
        ["yhat", "yhat_lower", "yhat_upper"]
    ].clip(lower=0)
    forecast_df["type"] = "forecast"

    temp_path = OUTPUT_PARQUET + ".tmp"
    forecast_df.to_parquet(temp_path, index=False)
    os.replace(temp_path, OUTPUT_PARQUET)
    log(f"[DONE] Forecasted {len(forecast_df)} days for {forecast_start.strftime('%B %Y')}")

    # ─── Save fitted values for entire training range ───
    fitted = model.predict(df[["ds"]])
    fitted_df = fitted[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fitted_df["type"] = "fitted"

    fitted_df.to_parquet(FITTED_PARQUET, index=False)
    log(f"[DONE] Saved fitted values for training range ({df['ds'].min().date()} to {df['ds'].max().date()})")

if __name__ == "__main__":
    forecast_and_save()
