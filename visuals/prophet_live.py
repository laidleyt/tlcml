import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import boto3
import os
from datetime import datetime


def download_forecast_from_s3():
    """
    Download the latest forecast parquet from S3.
    Always removes any old local version first.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_output.parquet"
    local_path = "data/forecast_output.parquet"

    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[DEBUG] Removed stale local forecast: {local_path}")

    s3_client.download_file(bucket_name, s3_key, local_path)
    print(f"[DEBUG] Downloaded {s3_key} to {local_path}")

    df_check = pd.read_parquet(local_path)
    print(f"[DEBUG] Forecast parquet range: {df_check['ds'].min()} to {df_check['ds'].max()}")


def download_input_from_s3():
    """
    Download the latest input parquet from S3.
    Always removes any old local version first.
    """
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_input.parquet"
    local_path = "data/forecast_input.parquet"

    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[DEBUG] Removed stale local input: {local_path}")

    s3_client.download_file(bucket_name, s3_key, local_path)
    print(f"[DEBUG] Downloaded {s3_key} to {local_path}")

    df_check = pd.read_parquet(local_path)
    print(f"[DEBUG] Input parquet range: {df_check['trip_date'].min()} to {df_check['trip_date'].max()}")


def make_live_forecast_figure():
    """
    Build the live forecast figure for the dashboard.
    """
    # Always pull fresh input AND forecast
    download_input_from_s3()
    download_forecast_from_s3()

    forecast_df = pd.read_parquet("data/forecast_output.parquet")
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    fitted_df = pd.read_parquet("data/forecast_fitted.parquet")
    fitted_df["ds"] = pd.to_datetime(fitted_df["ds"])

    actual_df = pd.read_parquet("data/forecast_input.parquet")
    actual_df["trip_date"] = pd.to_datetime(actual_df["trip_date"])

    last_actual = actual_df["trip_date"].max()
    forecast_start = (last_actual + pd.Timedelta(days=1)).replace(day=1)
    forecast_end = forecast_start + pd.offsets.MonthEnd(0)
    prev_month_start = forecast_start - relativedelta(months=1)

    print(f"[DEBUG] Last actual: {last_actual}")
    print(f"[DEBUG] Forecast window: {forecast_start} to {forecast_end}")

    filtered_df = forecast_df[
        (forecast_df["ds"] >= forecast_start) &
        (forecast_df["ds"] <= forecast_end)
    ]

    print(f"[DEBUG] Filtered forecast rows: {len(filtered_df)}")
    print(f"[DEBUG] Filtered forecast range: {filtered_df['ds'].min()} to {filtered_df['ds'].max()}")

    if filtered_df.empty:
        raise ValueError("[ERROR] Filtered forecast is empty! Check upstream forecast file.")

    display_start = (forecast_start - relativedelta(years=2)).strftime("%Y-%m-%d")
    display_end = (forecast_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    window_actuals = actual_df[
        (actual_df["trip_date"] >= forecast_start - relativedelta(months=1)) &
        (actual_df["trip_date"] <= last_actual)
    ]

    fitted_window = fitted_df[
        (fitted_df["ds"] >= forecast_start - relativedelta(months=1)) &
        (fitted_df["ds"] <= last_actual)
    ]

    merged_ci = pd.merge(
        fitted_window,
        window_actuals,
        left_on="ds", right_on="trip_date",
        how="inner"
    )
    merged_ci["in_ci"] = (
        (merged_ci["total_rides"] >= merged_ci["yhat_lower"]) &
        (merged_ci["total_rides"] <= merged_ci["yhat_upper"])
    )
    ci_hits = merged_ci["in_ci"].sum()
    ci_total = len(merged_ci)
    ci_pct = round((ci_hits / ci_total) * 100, 1) if ci_total > 0 else 0.0
    annotation_text = f"{ci_hits}/{ci_total} days ({ci_pct}%) within CI ({prev_month_start.strftime('%B %Y')})"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_upper"],
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), name="Forecast CI"
    ))

    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_upper"],
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat"],
        mode="lines", name="Forecast", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat"],
        mode="lines", name=None, showlegend=False, line=dict(color="blue")
    ))

    fig.add_trace(go.Scatter(
        x=actual_df["trip_date"], y=actual_df["total_rides"],
        mode="markers", name="Historical Actuals",
        marker=dict(size=2, color="black")
    ))
    fig.add_trace(go.Scatter(
        x=window_actuals["trip_date"], y=window_actuals["total_rides"],
        mode="markers", name="Actual Trips (Observed)",
        marker=dict(size=5, color="#FF6F00")
    ))

    fig.add_annotation(
        text=annotation_text,
        x=prev_month_start + pd.Timedelta(days=15),
        y=max(filtered_df["yhat_upper"].max(), fitted_df["yhat_upper"].max()) * 0.95,
        showarrow=False,
        font=dict(size=12),
        bgcolor="white"
    )

    fig.update_layout(
        xaxis=dict(range=[display_start, display_end]),
        yaxis=dict(title="Total Trips"),
        template="plotly_white",
        height=440
    )

    return fig
