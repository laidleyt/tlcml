import pandas as pd
from prophet import Prophet
from datetime import datetime
import os
import boto3

INPUT_PARQUET = "data/forecast_input.parquet"
OUTPUT_PARQUET = "data/forecast_output.parquet"

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def download_input_from_s3():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_input.parquet"

    s3_client.download_file(bucket_name, s3_key, INPUT_PARQUET)
    log(f"[PULL] Downloaded {s3_key} to {INPUT_PARQUET}")

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

def get_next_forecast_window(input_df, output_df):
    input_df["trip_date"] = pd.to_datetime(input_df["trip_date"])
    output_df["ds"] = pd.to_datetime(output_df["ds"]) if not output_df.empty else pd.Series(dtype="datetime64[ns]")

    latest_actual_date = input_df["trip_date"].max()

    latest_full_month_end = latest_actual_date.replace(day=1) + pd.offsets.MonthEnd(0)
    forecast_start = latest_full_month_end + pd.offsets.Day(1)
    forecast_end = forecast_start + pd.offsets.MonthEnd(0)

    log(f"[DEBUG] Latest actual date: {latest_actual_date}")
    log(f"[DEBUG] Latest full month end: {latest_full_month_end}")
    log(f"[DEBUG] Next forecast window: {forecast_start} to {forecast_end}")

    if not output_df.empty:
        latest_forecast_date = output_df["ds"].max()
        log(f"[DEBUG] Latest forecasted date: {latest_forecast_date}")

        if latest_forecast_date >= forecast_end:
            if latest_actual_date > latest_forecast_date:
                log("[FORCE] Actuals are newer than forecast. Forcing new forecast.")
                return forecast_start, forecast_end
            else:
                log("[DONE] Forecast is already up to date.")
                return None, None

    return forecast_start, forecast_end

def main():
    try:
        download_input_from_s3()
        while True:
            input_df = pd.read_parquet(INPUT_PARQUET)
            output_df = pd.read_parquet(OUTPUT_PARQUET) if os.path.exists(OUTPUT_PARQUET) else pd.DataFrame(columns=["ds"])

            forecast_start, forecast_end = get_next_forecast_window(input_df, output_df)

            if forecast_start is None:
                break

            log(f"[INFO] Forecasting {forecast_start.date()} to {forecast_end.date()}")

            train_df = input_df[
                (input_df["trip_date"] >= "2020-03-01") &
                (input_df["trip_date"] <= forecast_start - pd.offsets.Day(1))
            ].rename(columns={"trip_date": "ds", "total_rides": "y"}).dropna()

            model = Prophet()
            model.fit(train_df)

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

            df_all = pd.concat([output_df, forecast_window], ignore_index=True)
            df_all = df_all.drop_duplicates(subset="ds", keep="last").sort_values("ds")
            df_all.to_parquet(OUTPUT_PARQUET, index=False)

            upload_forecast_to_s3()
            log(f"[DONE] Appended forecast for {forecast_start.date()} to {forecast_end.date()}")
            break

    except Exception as e:
        log(f"[ERROR] Forecast failed: {e}")

if __name__ == "__main__":
    main()
