import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Load precomputed forecast from Parquet
merged = pd.read_parquet("data/bayes_forecast_placebo.parquet")

pio.renderers.default = 'browser'

def plot_orbit_forecast(df, title="Orbit Bayesian Forecast: January 10 Placebo"):
    intervention_date = pd.to_datetime('2022-01-09')
    inlay_df = df[(df['ds'] >= intervention_date) & (df['ds'] <= intervention_date + pd.Timedelta(days=30))]
    y_max = df['y'].max()

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='#BC13FE')))
    fig6.add_trace(go.Scatter(x=df['ds'], y=df['prediction'], mode='lines', name='Forecast', line=dict(color='#FF3131')))
    fig6.add_trace(go.Scatter(
        x=pd.concat([df['ds'], df['ds'][::-1]]),
        y=pd.concat([df['prediction_95'], df['prediction_5'][::-1]]),
        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='90% CI'
    ))

    fig6.add_vline(x=intervention_date, line_dash='dot', line_color='black')
    fig6.add_annotation(
        x=intervention_date,
        y=240000,
        text="January 10 Placebo",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        xshift=10,
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        opacity=1.0
    )

    fig6.add_trace(go.Scatter(
        x=inlay_df['ds'], y=inlay_df['y'], xaxis='x2', yaxis='y2',
        mode='lines', name='Actual (Inset)', line=dict(color='#BC13FE', width=1)
    ))
    fig6.add_trace(go.Scatter(
        x=inlay_df['ds'], y=inlay_df['prediction'], xaxis='x2', yaxis='y2',
        mode='lines', name='Forecast (Inset)', line=dict(color='#FF3131', width=1)
    ))
    fig6.add_trace(go.Scatter(
        x=pd.concat([inlay_df['ds'], inlay_df['ds'][::-1]]),
        y=pd.concat([inlay_df['prediction_95'], inlay_df['prediction_5'][::-1]]),
        xaxis='x2', yaxis='y2', fill='toself', fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip",
        name='90% CI (Inset)', showlegend=False
    ))

    fig6.update_layout(
        title=None,
        yaxis_title="Total Trips",
        template="plotly_white",
        legend=dict(
            orientation="h", yanchor="top", y=1.08, x=1.02, xanchor="right",
            bgcolor="white", bordercolor="lightgray", borderwidth=1,
            font=dict(size=18)
        ),
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(domain=[0, 1]),
        xaxis2=dict(domain=[0.40, 0.60], anchor='y2', showticklabels=True, tickangle=45, tickformat="%b %d", tickfont=dict(size=9)),
        yaxis2=dict(domain=[0.65, 0.95], anchor='x2', showgrid=True, linecolor='black', linewidth=1),
        height=480
    )
    fig6.add_shape(
        type="rect",
        xref="paper", yref="paper",
        x0=0.40, x1=0.60, y0=0.65, y1=0.95,
        line=dict(color="black", width=1),
        layer="above"
    )

    # Compute CI coverage stats (only for 30 days post-intervention)
    coverage_window = (df['ds'] >= intervention_date) & (df['ds'] < intervention_date + pd.Timedelta(days=30))
    within_ci = (
        (df['y'] >= df['prediction_5']) & (df['y'] <= df['prediction_95'])
    )
    ci_coverage = within_ci[coverage_window].sum()
    total_days = coverage_window.sum()

    fig6.add_annotation(
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        text=f"CI captured {ci_coverage} of {total_days} days (30-day window)",
        font=dict(size=11, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        opacity=0.95,
        xanchor="right",
        yanchor="bottom"
    )

    return fig6

fig6 = plot_orbit_forecast(merged)
