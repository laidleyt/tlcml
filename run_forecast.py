import pandas as pd
from prophet import Prophet
from datetime import datetime
import os
import boto3

INPUT_PARQUET = "data/forecast_input.parquet"
OUTPUT_PARQUET = "data/forecast_output.parquet"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def get_next_forecast_window(input_df, output_df):
    input_df["trip_date"] = pd.to_datetime(input_df["trip_date"])
    output_df["ds"] = pd.to_datetime(output_df["ds"]) if not output_df.empty else pd.Series(dtype="datetime64[ns]")

    latest_actual_date = input_df["trip_date"].max()

    today = datetime.today()
    if latest_actual_date.month == today.month and latest_actual_date.day < 28:
        latest_full_month_end = (latest_actual_date - pd.offsets.MonthBegin(1)).replace(day=1) + pd.offsets.MonthEnd(0)
    else:
        latest_full_month_end = latest_actual_date.replace(day=1) + pd.offsets.MonthEnd(0)

    forecast_start = latest_full_month_end + pd.offsets.Day(1)
    forecast_end = forecast_start + pd.offsets.MonthEnd(0)

    log(f"[DEBUG] Latest actual date: {latest_actual_date.date()}")
    log(f"[DEBUG] Latest full month end: {latest_full_month_end.date()}")
    log(f"[DEBUG] Next forecast window: {forecast_start.date()} to {forecast_end.date()}")

    if not output_df.empty:
        latest_forecast_date = output_df["ds"].max()
        if latest_forecast_date >= forecast_end:
            return None, None

    return forecast_start, forecast_end

def upload_forecast_to_s3():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_output.parquet"

    s3_client.upload_file(OUTPUT_PARQUET, bucket_name, s3_key)
    log(f"[PUSH] Uploaded {OUTPUT_PARQUET} to s3://{bucket_name}/{s3_key}")

def main():
    try:
        while True:
            input_df = pd.read_parquet(INPUT_PARQUET)
            output_df = pd.read_parquet(OUTPUT_PARQUET) if os.path.exists(OUTPUT_PARQUET) else pd.DataFrame(columns=["ds"])

            forecast_start, forecast_end = get_next_forecast_window(input_df, output_df)

            if forecast_start is None:
                log("[DONE] Forecast is already up to date.")
                break

            log(f"[INFO] Forecasting window: {forecast_start.date()} to {forecast_end.date()}")

            # Train Prophet
            train_df = input_df[
                (input_df["trip_date"] >= "2020-03-01") &
                (input_df["trip_date"] <= forecast_start - pd.offsets.Day(1))
            ].rename(columns={"trip_date": "ds", "total_rides": "y"})
            train_df = train_df.dropna()

            model = Prophet()
            model.fit(train_df)

            future = model.make_future_dataframe(
                periods=(forecast_end - forecast_start).days + 1,
                freq="D"
            )
            forecast = model.predict(future)

            # Extract only new window
            forecast_window = forecast[
                (forecast["ds"] >= forecast_start) &
                (forecast["ds"] <= forecast_end)
            ][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            forecast_window["type"] = "forecast"

            if os.path.exists(OUTPUT_PARQUET):
                df_existing = pd.read_parquet(OUTPUT_PARQUET)
                df_all = pd.concat([df_existing, forecast_window], ignore_index=True)
                df_all = df_all.drop_duplicates(subset="ds", keep="last")
                df_all = df_all.sort_values("ds")
            else:
                df_all = forecast_window

            df_all.to_parquet(OUTPUT_PARQUET, index=False)

            log(f"[DONE] Appended forecast for window: {forecast_start.date()} to {forecast_end.date()}")

    except Exception as e:
        log(f"[ERROR] Forecast failed: {e}")

if __name__ == "__main__":
    main()
    upload_forecast_to_s3()
