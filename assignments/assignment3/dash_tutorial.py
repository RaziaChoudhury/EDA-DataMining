"""
3 pillars

dash components - need to create a interactive capability
plotly graphs -
the callback - connects between the dash components and plotly graphs
            - a function that is automatically called by dash whenever an input
            component's property changes
"""

import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H1("testing", style={'text-align': 'center'}),
    dcc.Dropdown(
            options=[
                {'label': 'New York City', 'value': 'NYC'},
                {'label': 'Montréal', 'value': 'MTL'},
                {'label': 'San Francisco', 'value': 'SF'}
            ],
            value='MTL'
    )
])


if __name__ == "__main__":
    app.run_server(debug=True)