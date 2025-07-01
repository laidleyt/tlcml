import pandas as pd
import plotly.graph_objects as go
from dateutil.relativedelta import relativedelta
import calendar


def make_live_forecast_figure():
    # Load data
    forecast_df = pd.read_parquet("data/forecast_output.parquet")
    forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

    fitted_df = pd.read_parquet("data/forecast_fitted.parquet")
    fitted_df["ds"] = pd.to_datetime(fitted_df["ds"])

    actual_df = pd.read_parquet("data/forecast_input.parquet")
    actual_df["trip_date"] = pd.to_datetime(actual_df["trip_date"])

    # Compute window
    last_actual = actual_df["trip_date"].max()
    forecast_start = (last_actual + pd.Timedelta(days=1)).replace(day=1)
    forecast_end = forecast_start + pd.offsets.MonthEnd(0)
    prev_month_start = forecast_start - relativedelta(months=1)

    display_start = (forecast_start - relativedelta(years=2)).strftime("%Y-%m-%d")
    display_end = (forecast_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Slices
    window_actuals = actual_df[
        (actual_df["trip_date"] >= forecast_start - relativedelta(months=1)) &
        (actual_df["trip_date"] <= last_actual)
    ]

    fitted_window = fitted_df[
        (fitted_df["ds"] >= forecast_start - relativedelta(months=1)) &
        (fitted_df["ds"] <= last_actual)
    ]

    # CI annotation
    merged_ci = pd.merge(
        fitted_window, window_actuals,
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
        f"fell within forecast band ({prev_month_start.strftime('%B %Y')})."
    )

    fig = go.Figure()

    # Forecast CI
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), name='Forecast CI (80–95%)'
    ))

    # Fitted CI
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_upper"],
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat_lower"],
        fill='tonexty', fillcolor='rgba(150, 0, 255, 0.25)',
        line=dict(width=0), showlegend=False
    ))

    # Forecast & fitted lines
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"], y=forecast_df["yhat"],
        mode="lines", name="Forecast (Prophet)", line=dict(color="blue", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=fitted_df["ds"], y=fitted_df["yhat"],
        mode="lines", name=None, line=dict(color="blue", width=2), showlegend=False
    ))

    # Historical actuals
    fig.add_trace(go.Scatter(
        x=actual_df["trip_date"], y=actual_df["total_rides"],
        mode="markers", name="Historical Actuals",
        marker=dict(size=2, color="black", opacity=0.7),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=window_actuals["trip_date"], y=window_actuals["total_rides"],
        mode="markers",
        name=f"Actual Trips (Observed, {prev_month_start.strftime('%b %Y')})",
        marker=dict(size=5, color="#FF6F00"),
        hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
    ))

    # Guide lines
    fig.add_vrect(
        x0=forecast_start - relativedelta(months=1), x1=last_actual,
        fillcolor="lightgray", opacity=0.3, layer="below", line_width=0
    )
    fig.add_vline(x=forecast_start - relativedelta(months=1), line=dict(color="gray", dash="solid", width=1))
    fig.add_vline(x=forecast_start, line=dict(color="gray", dash="dot", width=1))
    fig.add_vline(x=forecast_end, line=dict(color="gray", dash="solid", width=1))

    # Annotation
    y_min = min(forecast_df["yhat_lower"].min(), fitted_df["yhat_lower"].min())
    y_max = max(forecast_df["yhat_upper"].max(), fitted_df["yhat_upper"].max())
    annotation_y = y_min + 0.33 * (y_max - y_min)

    fig.add_annotation(
        text=annotation_text,
        x=prev_month_start - pd.Timedelta(days=2),
        xref='x', y=annotation_y,
        showarrow=False, xanchor="right",
        font=dict(size=14), bgcolor="white",
        bordercolor="gray", borderwidth=1
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
