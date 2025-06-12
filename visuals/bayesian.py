import pandas as pd
import plotly.graph_objects as go

# ───────────── Load cached Orbit output ───────────── #
merged = pd.read_parquet("data/bayes_forecast_210.parquet")

# Uncomment the following block to retrain with Orbit (not used in production)
"""
import typing_extensions  # required by Orbit
from orbit.models import DLT
from sklearn.preprocessing import StandardScaler

# Load data
trips = pd.read_csv('data/daily_total_trips_patched.csv', parse_dates=['pickup_date'])
weather = pd.read_csv('data/max_daily_temperatures_2020_2022.csv', parse_dates=['datetime'])
mta = pd.read_csv('data/mta_weekly_subway.csv', parse_dates=['week_start'])
covid = pd.read_csv('data/covid19.csv', parse_dates=['date_of_interest'])

# Preprocess MTA data
mta_daily = mta.set_index('week_start').resample('D').ffill().reset_index()
mta_daily.rename(columns={'week_start': 'date', 'subway_rides': 'mta_ridership'}, inplace=True)
covid.rename(columns={'date_of_interest': 'date'}, inplace=True)

# Merge all data
trips.rename(columns={'pickup_date': 'date'}, inplace=True)
df = trips.merge(weather.rename(columns={'datetime': 'date'}), on='date', how='left')
df = df.merge(mta_daily, on='date', how='left')
df = df.merge(covid[['date', 'hospitalized']], on='date', how='left')
df = df[['date', 'total_trips', 'tempmax', 'mta_ridership', 'hospitalized']].dropna()
df.rename(columns={'date': 'ds', 'total_trips': 'y', 'hospitalized': 'covid_hospitalized'}, inplace=True)

# Standardize
scaler = StandardScaler()
regressors = ['tempmax', 'mta_ridership', 'covid_hospitalized']
df[regressors] = scaler.fit_transform(df[regressors])

# Split and train
train_df = df[(df['ds'] >= '2020-09-01') & (df['ds'] <= '2022-02-09')]
model = DLT(response_col='y', date_col='ds', regressor_col=regressors, seasonality=7, estimator='stan-mcmc', seed=8888)
model.fit(train_df)

# Predict
df_forecastable = df[df['ds'] >= '2020-09-01']
pred = model.predict(df_forecastable, include_ci=True)
merged = df_forecastable[['ds', 'y']].merge(pred, on='ds', how='left')

# merged.to_parquet("data/bayes_forecast_210.parquet")  # <–– Save once if regenerating
"""

# ───────────── Plot construction ───────────── #
def plot_orbit_forecast(df, title="Orbit Bayesian Forecast: NYS Mask Mandate Ended"):
    intervention_date = pd.to_datetime('2022-02-10')
    inlay_df = df[(df['ds'] >= intervention_date) & (df['ds'] <= intervention_date + pd.Timedelta(days=30))]

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual', line=dict(color='#BC13FE')))
    fig4.add_trace(go.Scatter(x=df['ds'], y=df['prediction'], mode='lines', name='Forecast', line=dict(color='#FF3131')))
    fig4.add_trace(go.Scatter(
        x=pd.concat([df['ds'], df['ds'][::-1]]),
        y=pd.concat([df['prediction_95'], df['prediction_5'][::-1]]),
        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', name='90% CI',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip"
    ))

    fig4.add_vline(x=intervention_date, line_dash='dot', line_color='black')
    fig4.add_annotation(
        x=intervention_date, y=240000,
        text="NYS Mask Mandate Lifted", showarrow=False,
        xanchor="left", yanchor="bottom", xshift=10,
        font=dict(size=12, color="black", family="Arial Black"),
        bgcolor="white", bordercolor="black", borderwidth=1, opacity=1.0
    )

    # Inset forecast
    fig4.add_trace(go.Scatter(x=inlay_df['ds'], y=inlay_df['y'], xaxis='x2', yaxis='y2',
                              mode='lines', name='Actual (Inset)', line=dict(color='#BC13FE', width=1)))
    fig4.add_trace(go.Scatter(x=inlay_df['ds'], y=inlay_df['prediction'], xaxis='x2', yaxis='y2',
                              mode='lines', name='Forecast (Inset)', line=dict(color='#FF3131', width=1)))
    fig4.add_trace(go.Scatter(
        x=pd.concat([inlay_df['ds'], inlay_df['ds'][::-1]]),
        y=pd.concat([inlay_df['prediction_95'], inlay_df['prediction_5'][::-1]]),
        xaxis='x2', yaxis='y2',
        fill='toself', fillcolor='rgba(255, 0, 0, 0.2)', name='90% CI (Inset)',
        line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
    ))

    # Coverage summary
    coverage_window = (df['ds'] >= intervention_date) & (df['ds'] < intervention_date + pd.Timedelta(days=30))
    within_ci = (df['y'] >= df['prediction_5']) & (df['y'] <= df['prediction_95'])
    ci_coverage = within_ci[coverage_window].sum()
    total_days = coverage_window.sum()
    fig4.add_annotation(
        xref="paper", yref="paper", x=0.99, y=0.01, showarrow=False,
        text=f"CI captured {ci_coverage} of {total_days} days (30-day window)",
        font=dict(size=11, color="black"),
        bgcolor="white", bordercolor="black", borderwidth=1, opacity=0.95,
        xanchor="right", yanchor="bottom"
    )

    fig4.update_layout(
        title=None,
        yaxis_title="Total Trips",
        template="plotly_white",
        height=480,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            x=1.02,
            xanchor="right",
            bgcolor="white",
            bordercolor="lightgray",
            borderwidth=1,
            font=dict(size=12)
        ),
        xaxis=dict(domain=[0, 1]),
        yaxis=dict(domain=[0, 1]),
        xaxis2=dict(domain=[0.40, 0.60], anchor='y2', showticklabels=True, tickangle=45, tickformat="%b %d", tickfont=dict(size=9)),
        yaxis2=dict(domain=[0.65, 0.95], anchor='x2', showgrid=True, linecolor='black', linewidth=1, showticklabels=True)
    )

    fig4.add_shape(
        type="rect", xref="paper", yref="paper",
        x0=0.40, x1=0.60, y0=0.65, y1=0.95,
        line=dict(color="black", width=1), layer="above"
    )

    return fig4

fig4 = plot_orbit_forecast(merged)
