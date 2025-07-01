import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.dependencies import ALL
import os

# Static visuals
from visuals.vis1 import fig as fig1
from visuals.vis2 import fig2
from visuals.prophet import fig3
from visuals.bayesian import fig4
from visuals.bayesian2 import fig5
from visuals.bayesian_placebo import fig6

# 🔑 LIVE figure generator (NO training)
from visuals.prophet_live import make_live_forecast_figure

# ────────────────────────────── #
# DASH APP SETUP
# ────────────────────────────── #

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "NYC Taxi ML Dashboard"

def serve_layout():
    return html.Div([
        html.Div([
            html.H2("NYC Taxi Rides: Live Forecasts, Anomalies & Counterfactuals", className="app-title"),
            dcc.Tabs(
                id='main-tab',
                value='forecast-tab',
                children=[
                    dcc.Tab(label='Forecasting', value='forecast-tab'),
                    dcc.Tab(label='Anomalies', value='anomaly-tab'),
                    dcc.Tab(label='Bayesian Counterfactuals', value='bayes-tab')
                ]
            ),
            html.Div(id='subtab-controls', className='subtab-container'),
            dcc.Store(id='subtab-store'),
            html.Div(id='visual-content', className='visual-container'),

            html.Div([
                html.Div("About", id="about-button", className="footer-tab", n_clicks=0),
                html.A("Repo", href="https://github.com/laidleyt/tlcml", target="_blank", className="footer-tab")
            ], className="footer-bar footer-center"),
        ], className="app-container"),

        dbc.Modal(
            id="about-modal",
            is_open=False,
            centered=True,
            backdrop=True,
            children=[
                dbc.ModalHeader(dbc.ModalTitle("About This Dashboard")),
                dbc.ModalBody([
                    html.P([
                        "This dashboard analyzes NYC Yellow Cab data from 2020–2025 using ML techniques like ",
                        html.A("Prophet", href="https://github.com/facebook/prophet", target="_blank"),
                        ", ",
                        html.A("DBSCAN", href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html", target="_blank"),
                        ", and ",
                        html.A("Orbit", href="https://github.com/uber/orbit", target="_blank"),
                        "."
                    ]),
                    html.P("It includes a live forecast where the latest NYC Yellow Cab data is ingested from the NYC TLC, appended to local storage, and used to project the following month."),
                    html.P("Anomaly detection focuses on in-city rides (excluding airports and suburbs) to identify behavioral patterns linked to social and public health changes."),
                    html.P("Finally, a Bayesian counterfactual forecasting model uses subway ridership, weather, and COVID hospitalizations as covariates to estimate how policy shifts affected taxi demand."),
                    html.P([
                        "Thanks to the ",
                        html.A("NYC TLC", href="https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page", target="_blank"),
                        " for making these data available."
                    ])
                ])
            ]
        ),
        dcc.Store(id='about-visible', data=False)
    ])

app.layout = serve_layout

# ────────────────────────────── #
# ABOUT MODAL
# ────────────────────────────── #

@app.callback(
    Output("about-modal", "is_open"),
    Input("about-button", "n_clicks"),
    State("about-modal", "is_open"),
    prevent_initial_call=True
)
def toggle_modal(n, is_open):
    return not is_open

# ────────────────────────────── #
# SUBTAB STORE INIT
# ────────────────────────────── #

@app.callback(
    Output('subtab-store', 'data'),
    Input('main-tab', 'value')
)
def initialize_subtab_value(main_tab):
    if main_tab == 'forecast-tab':
        return 'forecast-live'
    elif main_tab == 'anomaly-tab':
        return 'anomaly-overview'
    elif main_tab == 'bayes-tab':
        return 'bayes-210'
    return None

# ────────────────────────────── #
# VISUAL CONTENT CALLBACK
# ────────────────────────────── #

@app.callback(
    Output('visual-content', 'children'),
    Input('main-tab', 'value'),
    Input('subtab-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def update_visual(main_tab, subtab_value):
    def wrap_visual(fig, top_text):
        return html.Div([
            html.Div(
                dcc.Graph(
                    figure=fig,
                    className="unbound-plot",
                    style={"marginTop": "0px", "paddingTop": "0px"}
                ),
                className="graph-wrapper"
            ),
            html.Div(
                top_text,
                className="narration-box",
                style={"margin-bottom": "4px", "fontSize": "13px"}
            )
        ])

    if main_tab == 'forecast-tab':
        if subtab_value == 'forecast-live':
            fig = make_live_forecast_figure()
            top = [
                html.Strong("Current and Forecasted Taxi Trips: "),
                html.Span(
                    "This dynamically updated visual plots the newest available NYC taxi trip totals (in orange), "
                    "aggregated from individual trip data from NYC Open Data. In blue are predicted values obtained "
                    "using Prophet ML, trained on a time series back to March 2020. Historical actuals are "
                    "plotted in black, showing 24 months of prior activity."
                )
            ]
            return wrap_visual(fig, top)

        elif subtab_value == 'forecast-static':
            top = [
                html.Strong("Full Visualization of Time Series and Static Q1 2025 Prediction: "),
                html.Span(
                    "This visual shows the full time series back to March 2020, with predicted vs. actual values "
                    "for the first quarter of 2025."
                )
            ]
            return wrap_visual(fig3, top)

    elif main_tab == 'anomaly-tab':
        if subtab_value == 'anomaly-overview':
            top = [
                html.Strong("Anomaly detection overview (2020–2025): "),
                html.Span(
                    "This visual plots anomalous clusters identified with DBSCAN. Periods shaded in green or red "
                    "show high or low anomaly rates, with daily trip totals and COVID trends for context."
                )
            ]
            return wrap_visual(fig1, top)

        elif subtab_value == 'anomaly-zoom':
            top = [
                html.Strong("Detailed anomaly clusters (2022): "),
                html.Span(
                    "Focuses on 2022 to show post-COVID clusters, border reopenings, and pattern shifts."
                )
            ]
            return wrap_visual(fig2, top)

    elif main_tab == 'bayes-tab':
        if subtab_value == 'bayes-210':
            top = [
                html.Strong("Bayesian forecast: NYS mask mandate lifted (Feb 10, 2022): "),
                html.Span(
                    "Orbit-based counterfactual: estimates the impact of the state-level mask lift."
                )
            ]
            return wrap_visual(fig4, top)
        elif subtab_value == 'bayes-307':
            top = [
                html.Strong("Bayesian forecast: NYC Public Schools mask mandate lifted (Mar 7, 2022): "),
                html.Span(
                    "Shows the effect of the city schools mandate lift."
                )
            ]
            return wrap_visual(fig5, top)
        elif subtab_value == 'bayes-placebo':
            top = [
                html.Strong("Bayesian placebo test (Jan 10, 2022): "),
                html.Span(
                    "Validates no shift when no policy changed."
                )
            ]
            return wrap_visual(fig6, top)

    return html.Div("Invalid selection.")

# ────────────────────────────── #
# SUBTAB CONTROLS CALLBACK
# ────────────────────────────── #

@app.callback(
    Output('subtab-controls', 'children'),
    Input('main-tab', 'value'),
    Input('subtab-store', 'data')
)
def render_fake_radio_subtabs(main_tab, current_subtab):
    def make_div(label, value):
        selected = 'selected-tab' if current_subtab == value else ''
        return html.Div(
            label,
            className=f'subtab-item {selected}',
            n_clicks=0,
            id={'type': 'subtab-item', 'index': value}
        )

    if main_tab == 'forecast-tab':
        return html.Div([
            make_div("Live Forecast (Latest Month)", 'forecast-live'),
            make_div("Historic Forecast (2020-2025)", 'forecast-static')
        ], className='subtab-wrapper')
    elif main_tab == 'anomaly-tab':
        return html.Div([
            make_div("2020–2025 Overview", 'anomaly-overview'),
            make_div("2022 Cluster Analysis", 'anomaly-zoom')
        ], className='subtab-wrapper')
    elif main_tab == 'bayes-tab':
        return html.Div([
            make_div("Feb 10 (NYS Mask Lift)", 'bayes-210'),
            make_div("Mar 7 (NYCPS Mask Lift)", 'bayes-307'),
            make_div("Placebo (Jan 10)", 'bayes-placebo')
        ], className='subtab-wrapper')
    return html.Div()

# ────────────────────────────── #
# SUBTAB CLICK CALLBACK
# ────────────────────────────── #

@app.callback(
    Output('subtab-store', 'data', allow_duplicate=True),
    Input({'type': 'subtab-item', 'index': ALL}, 'n_clicks'),
    State({'type': 'subtab-item', 'index': ALL}, 'id'),
    prevent_initial_call=True
)
def set_selected_subtab(n_clicks_list, id_list):
    if not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    triggered_idx = n_clicks_list.index(max(n_clicks_list))
    return id_list[triggered_idx]['index']

# ────────────────────────────── #
# RUN SERVER
# ────────────────────────────── #

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port)
