# NYC Taxi ML Dashboard

This dashboard visualizes behavioral patterns in NYC Yellow Cab activity from 2020–2025 using machine learning models for time series forecasting, anomaly detection, and Bayesian counterfactual analysis.

### Features

- **Live Forecasting**  
  Ingests latest available data, summarizes/aggreagates trips by day, predicts upcoming monthly Yellow Cab ridership using Meta's Prophet model, trained on TLC trip data from March 2020 onward.

- **Anomaly Detection**  
  Identifies irregular trip patterns (excluding airport and suburban rides) using DBSCAN clustering, contextualized with COVID hospitalizations and ridership data.

- **Bayesian Counterfactuals**  
  Estimates the potential impact of policy changes (e.g., mask mandates) using Uber’s Orbit framework with covariates like weather, subway usage, and hospitalization trends.

### Data Source

All data are sourced from the [NYC Taxi and Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), with monthly Parquet files accessed via their AWS endpoint.

### Deployment

This app is built with Plotly Dash and is deployed via Render. It uses a scheduled ETL script to ingest new monthly data and automatically update forecasts.

### Repository Structure
├── app.py # Main Dash app with tab structure
├── visuals/ # All figures as separate modules
│ ├── prophet.py
│ ├── prophet_live.py
│ ├── vis1.py
│ ├── vis2.py
│ ├── bayesian.py
│ └── ...
├── assets/ # CSS styles
├── data/ # Preprocessed CSV/Parquet for plotting (Git-ignored)
├── update_ingest.py + run_forecast.py # Scheduled ETL + forecast update logic
├── requirements.txt
└── README.md


### License

This project is for demonstration purposes only and is not affiliated with the NYC TLC, Uber, Meta, or any third party.


