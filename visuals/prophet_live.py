import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import calendar
import boto3
import os
from datetime import datetime

def download_from_s3(bucket, s3_key, local_path):
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[DEBUG] Removed stale: {local_path}")
    s3_client.download_file(bucket, s3_key, local_path)
    print(f"[DEBUG] Downloaded s3://{bucket}/{s3_key} to {local_path}")

def make_live_forecast_figure():
    # Always pull fresh forecast + input
    bucket = "tlcml-forecast-data"
    download_from_s3(bucket, "forecast_output.parquet", "data/forecast_output.parquet")
    download_from_s3(bucket, "forecast_input.parquet", "data/forecast_input.parquet")

    # Read files
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

    if filtered_df.empty:
        raise ValueError("[ERROR] Filtered forecast is empty! Check upstream forecast file.")

    display_start = (forecast_start - relativedelta(years=2)).strftime("%Y-%m-%d")
    display_end = (forecast_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    window_actuals = actual_df[
        (actual_df["trip_date"] >= prev_month_start) &
        (actual_df["trip_date"] <= last_actual)
    ]

    fitted_window = fitted_df[
        (fitted_df["ds"] >= prev_month_start) &
        (fitted_df["ds"] <= last_actual)
    ]

    # Merge fitted + forecast to build continuous CI & line
    full_ci_df = pd.concat([fitted_df, filtered_df]).drop_duplicates(subset="ds").sort_values("ds")

    # CI hits for annotation
    merged_ci = pd.merge(
        fitted_window, window_actuals,
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
    annotation_text = (
        f"{ci_hits} of {ci_total} actual days ({ci_pct}%) "
        f"fell within forecast band ({prev_month_start.strftime('%B %Y')})."
    )

    fig = go.Figure()

    # Combined CI
    fig.add_trace(go.Scatter(
        x=full_ci_df["ds"], y=full_ci_df["yhat_upper"],
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=full_ci_df["ds"], y=full_ci_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), name="Forecast CI (80–95%)"
    ))

    # Combined line
    fig.add_trace(go.Scatter(
        x=full_ci_df["ds"], y=full_ci_df["yhat"],
        mode="lines", name="Forecast (Prophet)", line=dict(color="blue", width=2)
    ))

    # All historical actuals
    fig.add_trace(go.Scatter(
        x=actual_df["trip_date"], y=actual_df["total_rides"],
        mode="markers", name="Historical Actuals",
        marker=dict(size=2, color="black", opacity=0.7)
    ))

    # Recent observed window
    fig.add_trace(go.Scatter(
        x=window_actuals["trip_date"], y=window_actuals["total_rides"],
        mode="markers",
        name=f"Actual Trips (Observed, {prev_month_start.strftime('%b %Y')})",
        marker=dict(size=5, color="#FF6F00")
    ))

    # Guides & vrect
    fig.add_vrect(
        x0=prev_month_start, x1=last_actual,
        fillcolor="lightgray", opacity=0.3, layer="below", line_width=0
    )
    fig.add_vline(x=prev_month_start, line=dict(color="gray", dash="solid", width=1))
    fig.add_vline(x=forecast_start, line=dict(color="gray", dash="dot", width=1))
    fig.add_vline(x=forecast_end, line=dict(color="gray", dash="solid", width=1))

    # Annotation placement
    y_min = full_ci_df["yhat_lower"].min()
    y_max = full_ci_df["yhat_upper"].max()
    y_range = y_max - y_min
    annotation_y = y_min + 0.33 * y_range

    prev_month_label = calendar.month_abbr[prev_month_start.month]
    forecast_month_label = calendar.month_abbr[forecast_start.month]

    fig.add_annotation(
        text=annotation_text,
        x=prev_month_start - pd.Timedelta(days=2),
        xref='x', y=annotation_y,
        showarrow=False, xanchor="right",
        font=dict(size=14), bgcolor="white",
        bordercolor="gray", borderwidth=1
    )

    fig.add_annotation(
        text=prev_month_label,
        x=prev_month_start + pd.Timedelta(days=14),
        y=185000,
        showarrow=False,
        font=dict(size=14, color="gray"), opacity=0.9,
        xref="x", yref="y"
    )

    fig.add_annotation(
        text=forecast_month_label,
        x=forecast_start + pd.Timedelta(days=14),
        y=185000,
        showarrow=False,
        font=dict(size=14, color="gray"), opacity=0.9,
        xref="x", yref="y"
    )

    fig.update_layout(
        xaxis=dict(
            tickfont=dict(size=12), tickangle=45,
            range=[display_start, display_end]
        ),
        yaxis=dict(
            title=dict(text="Total Trips", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation="h", yanchor="top", y=0.96,
            xanchor="left", x=0.01,
            font=dict(size=11), bgcolor="white",
            bordercolor="lightgray", borderwidth=1
        ),
        template="plotly_white",
        height=440,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig
