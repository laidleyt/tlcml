import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Load and prep data
historical = pd.read_csv("data/daily_total_trips_patched.csv", parse_dates=["pickup_date"])
actual_q1_2025 = pd.read_csv("data/daily_total_trips_2025_Q1.csv", parse_dates=["pickup_date"])

train_df = historical[(historical["pickup_date"] >= "2020-01-01") & (historical["pickup_date"] < "2025-01-01")].copy()
train_df = train_df.rename(columns={"pickup_date": "ds", "total_trips": "y"})

# Prophet model
model = Prophet(daily_seasonality=True, yearly_seasonality=True)
model.fit(train_df)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Merge forecast + observed
actual_trim = actual_q1_2025.rename(columns={"pickup_date": "ds", "total_trips": "actual"})
merged = pd.merge(forecast, actual_trim, on="ds", how="left")

# Accuracy annotation
within_bounds = ((merged["actual"] >= merged["yhat_lower"]) & (merged["actual"] <= merged["yhat_upper"])).sum()
total_days = merged["actual"].notna().sum()
accuracy_text = f"{within_bounds} of {total_days} actual days ({within_bounds / total_days:.1%}) fell within forecast band."

fig3 = go.Figure()

# Forecast line with line break
fig3.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    mode='lines', name='Forecast\n(2020–2024 model)',
    line=dict(color='blue', width=2)
))


# Confidence interval
fig3.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat_upper"],
    mode='lines', line=dict(width=0),
    showlegend=False
))
fig3.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat_lower"],
    mode='lines', fill='tonexty',
    fillcolor='rgba(100,0,150,0.3)', line=dict(width=0),
    name='Forecast CI (80–95%)'
))

# Historical actuals
fig3.add_trace(go.Scatter(
    x=train_df["ds"], y=train_df["y"],
    mode='markers', name='Historical Actuals',
    marker=dict(color='black', size=2, opacity=0.5),
    hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
))

# 2025 actuals
fig3.add_trace(go.Scatter(
    x=merged["ds"], y=merged["actual"],
    mode='markers',
    name='Actual Trips (Observed, Jan–Mar 2025)',
    marker=dict(color='#FF6F00', size=4),
    hovertemplate="Date: %{x|%b %d, %Y}<br>Trips: %{y:,}<extra></extra>"
))

# Shading
fig3.add_vrect(
    x0="2025-01-01", x1="2025-03-31",
    fillcolor="lightgray", opacity=0.3,
    layer="below", line_width=0
)

# Accuracy annotation
fig3.add_annotation(
    x=0.95, y=0.19, xref="paper", yref="paper",
    text=accuracy_text,
    showarrow=False,
    font=dict(size=12),
    bgcolor="white", bordercolor="black", borderwidth=1, opacity=0.8
)

# Inset plot traces
forecast_jan_mar = forecast[forecast["ds"].between("2025-01-01", "2025-03-31")]
actual_jan_mar = merged[merged["ds"].between("2025-01-01", "2025-03-31")]

fig3.add_trace(go.Scatter(
    x=forecast_jan_mar["ds"], y=forecast_jan_mar["yhat_upper"],
    xaxis="x2", yaxis="y2",
    mode='lines', line=dict(width=0), showlegend=False
))
fig3.add_trace(go.Scatter(
    x=forecast_jan_mar["ds"], y=forecast_jan_mar["yhat_lower"],
    xaxis="x2", yaxis="y2",
    mode='lines', fill='tonexty',
    fillcolor='rgba(100,0,150,0.3)', line=dict(width=0), showlegend=False
))
fig3.add_trace(go.Scatter(
    x=forecast_jan_mar["ds"], y=forecast_jan_mar["yhat"],
    xaxis="x2", yaxis="y2",
    mode='lines', line=dict(color='blue', width=2), showlegend=False
))
fig3.add_trace(go.Scatter(
    x=actual_jan_mar["ds"], y=actual_jan_mar["actual"],
    xaxis="x2", yaxis="y2",
    mode='markers', marker=dict(color='#FF6F00', size=4), showlegend=False
))

# Inset bounding box
fig3.add_shape(
    type="rect", xref="paper", yref="paper",
    x0=0.75, x1=0.95, y0=0.79, y1=1.0,
    line=dict(color="black", width=1),
    fillcolor="rgba(255,255,255,0)", layer="above"
)

# Special points
special_points = pd.DataFrame({
    "ds": pd.to_datetime(["2025-02-14", "2025-03-29"]),
    "actual": [merged.loc[merged["ds"] == "2025-02-14", "actual"].values[0],
               merged.loc[merged["ds"] == "2025-03-29", "actual"].values[0]],
    "text": ["Feb 14, 2025 ❤️<br>Valentine’s Day spike in rides [158.6k]",
             "Mar 29, 2025 🌡️<br>Unseasonably warm 81°F day [163.1k]"]
})
fig3.add_trace(go.Scatter(
    x=special_points["ds"], y=special_points["actual"],
    xaxis="x2", yaxis="y2",
    mode='markers',
    marker=dict(color='#FF6F00', size=4),
    text=special_points["text"],
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))

# Arrows and reference lines
fig3.add_annotation(x="2025-02-15", y=164000, xref="x", yref="y", ax=0, ay=-40,
    axref="pixel", ayref="pixel", text="", showarrow=True,
    arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor="gray"
)
fig3.add_annotation(x="2025-03-29", y=168000, xref="x", yref="y", ax=0, ay=-40,
    axref="pixel", ayref="pixel", text="", showarrow=True,
    arrowhead=2, arrowsize=1.5, arrowwidth=1, arrowcolor="gray"
)
fig3.add_vline(x="2025-01-01", line=dict(color="black", width=1, dash='dot'))



# Axes and layout config
fig3.update_layout(
    xaxis=dict(domain=[0, 1], range=["2020-03-01", "2025-03-31"]),
    yaxis=dict(domain=[0.16, 1], range=[0, 260000]),
    xaxis2=dict(domain=[0.75, 0.95], anchor="y2", showticklabels=False),
    yaxis2=dict(domain=[0.82, 1.0], anchor="x2", range=[52000, 195000], showticklabels=False),
    legend=dict(
        orientation="v", yanchor="top", y=1.0,
        xanchor="left", x=0.05,
        font=dict(size=11),
        bgcolor="white", bordercolor="lightgray", borderwidth=1, tracegroupgap=3
    ),
    template="plotly_white",
    height=480,
    margin=dict(l=40, r=20, t=40, b=40)
)

# Uncomment to show or export
# fig3.show()
# fig3.write_image("output/prophet_forecast_static.png", scale=2)
