import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import calendar
import boto3
import os
from datetime import datetime

def log_debug(msg):
    """
    Log a debug message to stdout AND append to file for persistent trace.
    """
    log_line = f"{datetime.utcnow().isoformat()}Z | {msg}"
    print(log_line)
    with open("data/forecast_debug_log.txt", "a") as f:
        f.write(log_line + "\n")


def download_forecast_from_s3():
    """
    Download the forecast file fresh from S3. Always clear local first.
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
        log_debug(f"Removed existing local file: {local_path}")

    s3_client.download_file(bucket_name, s3_key, local_path)
    log_debug(f"Downloaded {s3_key} to {local_path}")

    df_check = pd.read_parquet(local_path)
    log_debug(f"S3 parquet check: min={df_check['ds'].min()} max={df_check['ds'].max()}")


def make_live_forecast_figure():
    """
    Generates the live Prophet forecast figure.
    """
    download_forecast_from_s3()

    forecast_df = pd.read_parquet("data/forecast_output.parquet")
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    fitted_df = pd.read_parquet("data/forecast_fitted.parquet")
    fitted_df["ds"] = pd.to_datetime(fitted_df["ds"])

    actual_df = pd.read_parquet("data/forecast_input.parquet")
    actual_df["trip_date"] = pd.to_datetime(actual_df["trip_date"])

    last_actual = actual_df["trip_date"].max()
    forecast_start = (last_actual + pd.Timedelta(days=1)).replace(day=1)
    forecast_end = (forecast_start + relativedelta(months=1)) - pd.Timedelta(days=1)

    prev_month_start = forecast_start - relativedelta(months=1)

    # ── DEBUG snapshot ───────────────────────────────
    raw_min, raw_max = forecast_df['ds'].min(), forecast_df['ds'].max()
    log_debug(f"Raw forecast window: min={raw_min}, max={raw_max}")
    log_debug(f"Forecast Start: {forecast_start}, Forecast End: {forecast_end}")

    filtered_df = forecast_df[
        (forecast_df["ds"] >= forecast_start) &
        (forecast_df["ds"] <= forecast_end)
    ]

    filtered_min, filtered_max = filtered_df['ds'].min(), filtered_df['ds'].max()
    log_debug(f"Filtered forecast window: min={filtered_min}, max={filtered_max}")

    if filtered_df.empty:
        err_msg = (
            f"[FAILSAFE] Filtered forecast is empty! "
            f"Raw min={raw_min}, max={raw_max}; "
            f"Expected: {forecast_start} → {forecast_end}"
        )
        log_debug(err_msg)
        raise ValueError(err_msg)

    # Also write snapshot to file for permanent record
    snapshot = (
        f"Run at: {datetime.utcnow().isoformat()}Z\n"
        f"Raw parquet: min={raw_min}, max={raw_max}\n"
        f"Filtered: min={filtered_min}, max={filtered_max}\n"
        f"Window: forecast_start={forecast_start}, forecast_end={forecast_end}\n"
    )
    with open("data/debug_forecast_snapshot.txt", "w") as f:
        f.write(snapshot)

    # ── Standard figure code ───────────────────────────────

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
        left_on="ds", right_on="trip_date", how="inner"
    )
    merged_ci["in_ci"] = (
        (merged_ci["total_rides"] >= merged_ci["yhat_lower"]) &
        (merged_ci["total_rides"] <= merged_ci["yhat_upper"])
    )
    ci_hits = merged_ci["in_ci"].sum()
    ci_total = len(merged_ci)
    ci_pct = round((ci_hits / ci_total) * 100, 1) if ci_total > 0 else 0.0

    annotation_text = (
        f"{ci_hits} of {ci_total} days ({ci_pct}%) within CI "
        f"({(forecast_start - relativedelta(months=1)).strftime('%B %Y')})"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_upper"],
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), name='Forecast CI'
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
        mode="lines", showlegend=False, line=dict(color="blue")
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

    fig.update_layout(
        xaxis=dict(range=[display_start, display_end]),
        yaxis=dict(title="Total Trips"),
        template="plotly_white",
        height=440
    )

    return fig
