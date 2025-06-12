
################################################################################################# plot anomalies
import pandas as pd
import plotly.graph_objects as go
import numpy as np


# Load total trip counts
daily_totals = pd.read_csv("data/daily_total_trips_patched.csv", parse_dates=["pickup_date"])

# Load your datasets
anomalies = pd.read_csv("data/anomalies_clustered_2022.csv", parse_dates=["pickup_datetime"])
clusters = pd.read_csv("data/filtered_clusters_condensed_2022.csv", parse_dates=["start_date", "end_date"])

# Aggregate daily anomaly count
anomalies['pickup_date'] = anomalies['pickup_datetime'].dt.date
daily = anomalies.groupby('pickup_date').size().reset_index(name='n_anomalies')
daily['pickup_date'] = pd.to_datetime(daily['pickup_date'])
daily['rolling_7'] = daily['n_anomalies'].rolling(window=7, center=True).mean()
daily = daily.merge(daily_totals, on="pickup_date", how="left")

daily['anomaly_rate'] = daily['n_anomalies'] / daily['total_trips']
mean_rate = daily['anomaly_rate'].mean()
std_rate = daily['anomaly_rate'].std()
daily['anomaly_z'] = (daily['anomaly_rate'] - mean_rate) / std_rate

cluster_scores = []

# Filter to 2022
anomalies = anomalies[(anomalies['pickup_datetime'] >= "2022-01-01") & (anomalies['pickup_datetime'] < "2023-01-01")].copy()
daily_totals = daily_totals[(daily_totals['pickup_date'] >= "2022-01-01") & (daily_totals['pickup_date'] < "2023-01-01")].copy()


for _, row in clusters.iterrows():
    mask = (daily['pickup_date'] >= row['start_date']) & (daily['pickup_date'] <= row['end_date'])
    avg_z = daily.loc[mask, 'anomaly_z'].mean()
    cluster_scores.append(avg_z)

clusters['avg_anomaly_z'] = cluster_scores

# Compute composite Z scores (distance + fare) for each cluster
def compute_composite_z(row, all_rides):
    cluster_mask = (anomalies['pickup_datetime'] >= row['start_date']) & (anomalies['pickup_datetime'] <= row['end_date'])
    cluster_rides = anomalies[cluster_mask]

    # Global stats
    mean_dist = all_rides['trip_distance'].mean()
    std_dist = all_rides['trip_distance'].std()
    mean_fare = all_rides['fare_amount'].mean()
    std_fare = all_rides['fare_amount'].std()

    # Cluster stats
    z_dist = ((cluster_rides['trip_distance'].mean() - mean_dist) / std_dist) if std_dist > 0 else 0
    z_fare = ((cluster_rides['fare_amount'].mean() - mean_fare) / std_fare) if std_fare > 0 else 0

    return (z_dist + z_fare) / 2

# Apply to cluster DataFrame
clusters['avg_composite_z'] = clusters.apply(lambda row: compute_composite_z(row, anomalies), axis=1)


# Load and process COVID data

covid = pd.read_csv("data/covid19.csv", parse_dates=["date_of_interest"])
covid = covid.rename(columns={"date_of_interest": "date", "HOSPITALIZED_COUNT": "hospitalizations"})

# Load MTA ridership data and merge with daily trips

mta = pd.read_csv("data/mta_weekly_subway.csv", parse_dates=["week_start"])
mta = mta.rename(columns={"week_start": "date", "subway_rides": "weekly_subway_rides"})
mta = mta.sort_values("date").reset_index(drop=True)

covid = covid[(covid['date'] >= "2022-01-01") & (covid['date'] < "2023-01-01")].copy()
mta = mta[(mta['date'] >= "2022-01-01") & (mta['date'] < "2023-01-01")].copy()


daily = pd.merge_asof(
    daily.sort_values("pickup_date"),
    mta.sort_values("date"),
    left_on="pickup_date",
    right_on="date",
    direction="backward"
)

# Create plot
fig2 = go.Figure()

# # Daily count line
# fig2.add_trace(go.Scatter(
#     x=daily['pickup_date'],
#     y=daily['n_anomalies'],
#     mode='lines',
#     name='Anomalies',
#     line=dict(color='royalblue'),
#     hovertemplate="%{x|%b %d, %Y}<br>%{y:,} Daily Anomalous Trips<extra></extra>"
# ))

# Cluster shading with hover
y_max = daily['n_anomalies'].max()

for i, row in clusters.iterrows():
    fillcolor = 'rgba(0, 200, 0, 0.2)' if row['avg_composite_z'] >= 0 else 'rgba(255, 0, 0, 0.2)'

    fig2.add_vrect(
        x0=row['start_date'],
        x1=row['end_date'],
        fillcolor=fillcolor,
        opacity=0.95,
        layer='below',
        line_width=0
    )

    fig2.add_annotation(
        x=row['start_date'] + (row['end_date'] - row['start_date']) / 2, 
        y=y_max * 0.3,
        text=f"{row['start_date'].strftime('%b %d').upper()}–{row['end_date'].strftime('%b %d').upper()}",
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=12, color="black"),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        opacity=0.95
    )

# First, round each date to the first day of the month
mta['month'] = mta['date'].dt.to_period('M').dt.to_timestamp()

# Aggregate by month (or quarter if needed)
monthly = mta.groupby('month')['weekly_subway_rides'].mean().reset_index()

# Normalize for plotting in paper space
max_rides = monthly['weekly_subway_rides'].max()

fig2.add_trace(go.Scatter(
    x=daily['pickup_date'],
    y=daily['total_trips'],
    name='Trips',
    yaxis='y2',
    line=dict(color='#BC13FE', width=2, dash='dot'),
    hovertemplate="%{x|%b %d, %Y}<br>%{y:,} total taxi trips<extra></extra>"
))

# Daily count line
fig2.add_trace(go.Scatter(
    x=daily['pickup_date'],
    y=daily['n_anomalies'],
    name="Anomalies",
    mode="lines",
    line=dict(color="#0077B6"),
    fill="tozeroy",
    fillcolor="rgba(0, 119, 182, 0.25)",
    hovertemplate="%{x|%b %d, %Y}<br>%{y:,} Daily Anomalous Trips<extra></extra>"
))

fig2.add_trace(go.Scatter(
    x=covid['date'],
    y=covid['hospitalizations'],
    name="Covid⚕️",
    yaxis='y3',
    line=dict(color='#FF3131', width=2),
    opacity=0.8,
    hovertemplate="%{x|%b %d, %Y}<br>%{y:,} COVID hospitalizations<extra></extra>"
))

fig2.add_shape(
    type="line",
    x0="2022-02-10", x1="2022-02-10",
    y0=0, y1=1,
    yref='paper',
    line=dict(color="black", width=3, dash="dot"),
    layer="above"
)

fig2.add_shape(
    type="line",
    x0="2022-03-07", x1="2022-03-07",
    y0=0, y1=1,
    yref='paper',
    line=dict(color="black", width=3, dash="dot"),
    layer="above"
)

fig2.add_annotation(
    x="2022-02-10",
    y=y_max * 1.49,
    text="End NYS Mask",
    showarrow=False,
    font=dict(size=12, color="#000000"),
    bgcolor="white",
    bordercolor="black",       
    borderwidth=1,             
    opacity=0.95,
    xanchor="right",
    yanchor="bottom"
)

fig2.add_annotation(
    x="2022-03-07",
    y=y_max * 1.39,
    text="End NYCPS Mask",
    showarrow=False,
    font=dict(size=12, color="#000000"),
    bgcolor="white",
    bordercolor="black",       
    borderwidth=1,             
    opacity=0.95,
    xanchor="right",
    yanchor="bottom"
)

fig2.add_annotation(
    x=1.02,  
    y=0.5,   
    xref='paper',
    yref='paper',
    text='Daily Trips (purple) & COVID hosp. (red)',
    showarrow=False,
    font=dict(size=12, color='black'),
    align='center',
    textangle=90,
    xanchor="left",
    yanchor="middle"
)

fig2.update_layout(
    title=None,  
    xaxis=dict(
        range=[daily['pickup_date'].min(), daily['pickup_date'].max()],
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        dtick="M1",
        tickformat="%b",
        ticklabelmode="period"
    ),
    yaxis=dict(
        title="Anomalous Trips",
        showgrid=True,
        gridcolor="lightgray",
        gridwidth=1,
        range=[0,500]
    ),
    yaxis2=dict(
    title="",  # blank
    overlaying='y',
    side='right',
    showgrid=False,
    tickfont=dict(size=10, color='#BC13FE'),
    ticklabelposition="outside right",
    ticklabeloverflow="allow",
    range=[0, 180000],        
    tickvals=[90000, 180000],  
    ticktext=["90k", "180k"]  
    ),
    yaxis3=dict(
    overlaying='y',
    side='right',
    position=1.0,
    showgrid=False,
    tickvals=[25, 250],
    tickfont=dict(size=11, color='#FF3131'),
    ticklabelposition="outside right",
    ticklabeloverflow="allow"
    ),
    legend=dict(
        title="",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1, 
        font=dict(size=12),
        bgcolor="white",              # background box
        bordercolor="lightgray",      # border color
        borderwidth=1   
    ),
    template="plotly_white",
    height=460
)

