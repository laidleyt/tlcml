import pandas as pd
from prophet import Prophet
from datetime import datetime
import os
import boto3

INPUT_PARQUET = "data/forecast_input.parquet"
OUTPUT_PARQUET = "data/forecast_output.parquet"
FITTED_PARQUET = "data/forecast_fitted.parquet"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_next_forecast_window(input_df):
    input_df["trip_date"] = pd.to_datetime(input_df["trip_date"])
    latest_actual_date = input_df["trip_date"].max()

    today = datetime.today()
    if latest_actual_date.month == today.month and latest_actual_date.day < 28:
        latest_full_month_end = (latest_actual_date - pd.offsets.MonthBegin(1)).replace(day=1) + pd.offsets.MonthEnd(0)
    else:
        latest_full_month_end = latest_actual_date.replace(day=1) + pd.offsets.MonthEnd(0)

    forecast_start = latest_full_month_end + pd.offsets.Day(1)
    forecast_end = forecast_start + pd.offsets.MonthEnd(0)

    log(f"Latest actual date: {latest_actual_date.date()}")
    log(f"Latest full month end: {latest_full_month_end.date()}")
    log(f"Next forecast window: {forecast_start.date()} to {forecast_end.date()}")

    return forecast_start, forecast_end

def download_input_from_s3():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_input.parquet"

    s3_client.download_file(bucket_name, s3_key, INPUT_PARQUET)
    log(f"Downloaded s3://{bucket_name}/{s3_key} to {INPUT_PARQUET}")

def upload_forecast_to_s3():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"

    s3_client.upload_file(OUTPUT_PARQUET, bucket_name, "forecast_output.parquet")
    log(f"Uploaded {OUTPUT_PARQUET} to s3://{bucket_name}/forecast_output.parquet")

    s3_client.upload_file(FITTED_PARQUET, bucket_name, "forecast_fitted.parquet")
    log(f"Uploaded {FITTED_PARQUET} to s3://{bucket_name}/forecast_fitted.parquet")

def main():
    try:
        download_input_from_s3()

        input_df = pd.read_parquet(INPUT_PARQUET)
        forecast_start, forecast_end = get_next_forecast_window(input_df)

        train_df = input_df[
            (input_df["trip_date"] >= "2020-03-01") &
            (input_df["trip_date"] <= forecast_start - pd.offsets.Day(1))
        ].rename(columns={"trip_date": "ds", "total_rides": "y"}).dropna()

        log(f"Training rows: {len(train_df)}")
        model = Prophet()
        model.fit(train_df)

        # Fitted values for training period
        fitted = model.predict(train_df[["ds"]])
        fitted = fitted[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        fitted["type"] = "fitted"
        fitted.to_parquet(FITTED_PARQUET, index=False)
        log(f"Saved fitted values to {FITTED_PARQUET}")

        # Forecast window
        future = model.make_future_dataframe(
            periods=(forecast_end - forecast_start).days + 1,
            freq="D"
        )
        forecast = model.predict(future)

        forecast_window = forecast[
            (forecast["ds"] >= forecast_start) &
            (forecast["ds"] <= forecast_end)
        ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        forecast_window["type"] = "forecast"

        log(f"Forecast window row count: {len(forecast_window)}")
        log(f"Forecast window dates: {forecast_window['ds'].min().date()} to {forecast_window['ds'].max().date()}")

        forecast_window.to_parquet(OUTPUT_PARQUET, index=False)
        log(f"Saved forecast to {OUTPUT_PARQUET}")

        upload_forecast_to_s3()
        log(f"Forecast complete: {forecast_start.date()} to {forecast_end.date()}")

    except Exception as e:
        log(f"[ERROR] Forecast failed: {e}")

if __name__ == "__main__":
    main()
