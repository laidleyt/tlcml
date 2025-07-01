import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import calendar
import boto3
import os
from datetime import datetime

def download_forecast_from_s3():
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    bucket_name = "tlcml-forecast-data"
    s3_key = "forecast_output.parquet"
    local_path = "data/forecast_output.parquet"

    # 🚫 Always clear old local copy to avoid stale read
    if os.path.exists(local_path):
        os.remove(local_path)
        print(f"[DEBUG] Removed existing local file: {local_path}")

    s3_client.download_file(bucket_name, s3_key, local_path)
    print(f"[PULL] Downloaded {s3_key} to {local_path}")

    # ✅ Double check the window from S3
    df_check = pd.read_parquet(local_path)
    print(f"[DEBUG] Downloaded file min={df_check['ds'].min()} max={df_check['ds'].max()}")


def make_live_forecast_figure():
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

    raw_min = forecast_df['ds'].min()
    raw_max = forecast_df['ds'].max()
    print(f"[DEBUG] Raw forecast parquet window: min={raw_min} max={raw_max}")

    filtered_df = forecast_df[
        (forecast_df["ds"] >= forecast_start) &
        (forecast_df["ds"] <= forecast_end)
    ]

    filtered_min = filtered_df['ds'].min()
    filtered_max = filtered_df['ds'].max()
    print(f"[DEBUG] Filtered forecast_df window: min={filtered_min} max={filtered_max}")
    print(f"[DEBUG] Forecast Start: {forecast_start}, Forecast End: {forecast_end}")

    if filtered_df.empty:
        raise ValueError(
            f"[FAILSAFE] Filtered forecast_df is empty!\n"
            f"Raw parquet min: {raw_min} max: {raw_max}\n"
            f"Requested window: {forecast_start} to {forecast_end}"
        )

    snapshot = (
        f"Run Timestamp: {datetime.utcnow().isoformat()}Z\n"
        f"Raw forecast parquet: min={raw_min}, max={raw_max}\n"
        f"Filtered forecast parquet: min={filtered_min}, max={filtered_max}\n"
        f"Expected window: forecast_start={forecast_start}, forecast_end={forecast_end}\n"
    )
    with open("data/debug_forecast_snapshot.txt", "w") as f:
        f.write(snapshot)

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
        fitted_window, actual_df[
            (actual_df["trip_date"] >= forecast_start - relativedelta(months=1)) &
            (actual_df["trip_date"] <= last_actual)
        ],
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
        f"{ci_hits} of {ci_total} actual days ({ci_pct}%) "
        f"fell within forecast band ({(forecast_start - relativedelta(months=1)).strftime('%B %Y')})."
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), name='Forecast CI (80–95%)'
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=filtered_df["ds"], y=filtered_df["yhat"],
        mode="lines", name="Forecast (Prophet)", line=dict(color="blue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat"],
        mode="lines", name=None, line=dict(color="blue", width=2), showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=actual_df["trip_date"], y=actual_df["total_rides"],
        mode="markers", name="Historical Actuals",
        marker=dict(size=2, color="black", opacity=0.7),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=window_actuals["trip_date"], y=window_actuals["total_rides"],
        mode="markers",
        name=f"Actual Trips (Observed, {(forecast_start - relativedelta(months=1)).strftime('%b %Y')})",
        marker=dict(size=5, color="#FF6F00"),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
    ))

    fig.add_vrect(
        x0=forecast_start - relativedelta(months=1), x1=last_actual,
        fillcolor="lightgray", opacity=0.3, layer="below", line_width=0
    )
    fig.add_vline(x=forecast_start - relativedelta(months=1), line=dict(color="gray", dash="solid", width=1))
    fig.add_vline(x=forecast_start, line=dict(color="gray", dash="dot", width=1))
    fig.add_vline(x=forecast_end, line=dict(color="gray", dash="solid", width=1))

    y_min = min(filtered_df["yhat_lower"].min(), fitted_df["yhat_lower"].min())
    y_max = max(filtered_df["yhat_upper"].max(), fitted_df["yhat_upper"].max())
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
        font=dict(size=14, color="gray"),
        opacity=0.9,
        xref="x", yref="y"
    )

    fig.add_annotation(
        text=forecast_month_label,
        x=forecast_start + pd.Timedelta(days=14),
        y=185000,
        showarrow=False,
        font=dict(size=14, color="gray"),
        opacity=0.9,
        xref="x", yref="y"
    )

    fig.update_layout(
        xaxis=dict(tickfont=dict(size=12), tickangle=45, range=[display_start, display_end]),
        yaxis=dict(title=dict(text="Total Trips", font=dict(size=14)),
                   tickfont=dict(size=12), range=[30000, 190000]),
        legend=dict(orientation="h", yanchor="top", y=0.96,
                    xanchor="left", x=0.01,
                    font=dict(size=11), bgcolor="white",
                    bordercolor="lightgray", borderwidth=1),
        template="plotly_white",
        height=440,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig
