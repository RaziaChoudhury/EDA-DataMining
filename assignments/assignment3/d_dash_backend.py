from typing import Tuple

import dash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Output, Input, State
from matplotlib.widgets import Button, Slider
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

from assignments.assignment1.a_load_file import *
from assignments.assignment1.e_experimentation import *
from assignments.assignment3.a_libraries import *
from assignments.assignment3.b_simple_usages import *
##############################################
# Now let's use dash, a library built on top of flask (a backend framework for python) and plotly
# Check the documentation at https://dash.plotly.com/
# For a complete example, check https://dash-bootstrap-components.opensource.faculty.ai/examples/iris/
# Example(s). Read the comments in the following method(s)
##############################################
def dash_simple_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    The focus is to create a fig from plotly and add it to dash, but differently from using just plotly, now we can use html elements,
    such as H1 for headers, Div for dividers, and all interations (buttons, sliders, etc).
    Check dash documentation for html and core components.
    """
    app = dash.Dash(__name__)

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    # You create a fig just as you did in a_
    fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        dcc.Graph(
            id='example-graph',
            figure=fig  # and include the fig here as a dcc.Graph
        )
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_with_bootstrap_example():
    """
    Here is a simple example from https://dash.plotly.com/layout
    See examples of components from the bootstrap library at https://dash-bootstrap-components.opensource.faculty.ai/docs/components/alert/
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    fig1 = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
    fig2 = px.line(df, x="Fruit", y="Amount", color="City")

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.Div(children='Dash: A web application framework for Python.'), md=4),
            dbc.Col(dbc.Button('Example Button', color='primary', style={'margin-bottom': '1em'}, block=True), md=8)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph1', figure=fig1)),
            dbc.Col(dcc.Graph(id='example-graph2', figure=fig2))
        ])
    ])

    # To run the app, use app.run_server(debug=True)
    # However, do return the app instead so that the file can be run as in the end of this file
    return app


def dash_callback_example():
    """
    Here is a more complex example that uses callbacks. With this example, I believe you will suddenly perceive why dash (and webapps)
    are so much better for visual analysis.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    df = pd.DataFrame({
        "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
        "Amount": [4, 1, 2, 2, 4, 5],
        "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    })

    app.layout = dbc.Container([
        html.H1(children='Hello Dash'),
        html.Div(children='Dash: A web application framework for Python.'),
        html.Hr(),
        dbc.FormGroup([
            dbc.Label("Choose Dataset"),
            dcc.Dropdown(id="dropdown", value=1, options=[{"label": "First Data", "value": 1}, {"label": "Second Data", "value": 2}]),
        ]),
        dbc.FormGroup([
            dbc.Label(id='slider-value'),
            dcc.Slider(id="slider", min=1, max=10, step=0.5, value=1),
        ]),
        dbc.Button('Run Callback', id='example-button', color='primary', style={'margin-bottom': '1em'}, block=True),
        dbc.Row([
            dbc.Col(dcc.Graph(id='example-graph')),  # Not including fig here because it will be generated with the callback
        ])
    ])

    @app.callback(  # See documentation or tutorial to see how to use this
        Output('example-graph', 'figure'),  # Outputs is what you wish to update with the callback, which in this case is the figure
        [Input('example-button', 'n_clicks')],  # Use inputs to define when this callback is called, and read from the values in the inputs as parameters in the method
        [State('dropdown', 'value'),  # Use states to read values from the interface, but values only in states will not trigger the callback when changed
         State('slider', 'value')])  # For example, here if you change the slider, this method will not be called, it will only be called when you click the button
    def update_figure(n_clicks, dropdown_value, slider_value):
        df2 = df[:]
        df2.Amount = df2.Amount * slider_value
        if dropdown_value == 1:
            return px.bar(df2, x="Fruit", y="Amount", color="City", barmode="group")
        else:
            return px.line(df2, x="City", y="Amount", color="Fruit")

    @app.callback(Output('slider-value', 'children'), [Input('slider', 'value')])
    def update_slider_value(slider):
        return f'Multiplier: {slider}'

    #  You can also use app.callback to get selection from any of the plotly graphs, including tables and maps, and update anything you wish.
    #  See some examples at https://dash-gallery.plotly.host/Portal/

    return app


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def dash_task():
    """
    There is only only one task to do, a web app with:
    1. Some nice title
    2. One visualization placeholder for dataset visualization
        a. A dropdown to allow me to select which dataset I want to see (iris, video_game and life_expectancy)
        b. Two other dropdowns for me to choose what column to put in x and what column to put in y of the visualization
        c. Another dropdown for me to choose what type of graph I want (see examples in file a_) (at least 3 choices of graphs)
        d. Feel free to change the structure of the dataset if you prefer (e.g. change life_expectancy so that
            there is one column of "year", one for "country" and one for "value")
    4. A https://dash-bootstrap-components.opensource.faculty.ai/docs/components/card/ with the number of rows being showed on the above graph
    5. Another visualization with:
        a. It will containing the figure created in the tasks in a_, b_ or c_ related to plotly's figures
        b. Add a dropdown for me to choose among 3 (or more if you wish) different graphs from a_, b_ or c_ (choose the ones you like)
        c. In this visualization, if I select data in the visualization, update some text in the page (can be a new bootstrap card with text inside)
            with the number of values selected. (see https://dash.plotly.com/interactive-graphing for examples)
    """
    app = dash.Dash(__name__,
                    external_stylesheets=[dbc.themes.BOOTSTRAP])
    datasets = {
        "Iris": read_dataset(Path('..', '..', 'iris.csv'))[["sepal_length",  "sepal_width",  "petal_length",  "petal_width"]],
        "Video Game": process_amazon_video_game_dataset()[["count", "review", "asin"]],
        "Life Expectancy": process_life_expectancy_dataset_lon()[["year", "country", "value"]]
    }
    app.layout = html.Div([
        html.H1("Visual Analytics - A3", style={'text-align': 'center'}),
        dcc.Dropdown(
            id='dataset_selection',
            options=[
                {'label': 'Iris', 'value': "Iris"},
                {'label': 'Video Game', 'value': "Video Game"},
                {'label': 'Life Expectancy', 'value': "Life Expectancy"}
            ],
            multi=False,
            value="Iris"
        ),
        dcc.Dropdown(
            id='x_selection',
            options=[{"label":c, "value":c} for c in datasets["Iris"].columns],
            value='sepal_width',
            multi=False
        ),
        dcc.Dropdown(
            id='y_selection',
            options=[{"label":c, "value":c} for c in datasets["Iris"].columns],
            value='sepal_length',
            multi=False
        ),
        dcc.Dropdown(
            id="graph_type",
            options = [
                {'label' : 'Bar Chart', 'value': "bar"},
                {'label': 'Scatter Plot', 'value': "scatter"},
                {'label': 'Line Plot', 'value': "line"}
            ],
            value='bar'
        ),
        # note button added to allow user to make selections and then plot graph
        dbc.Button('Plot Graph', id='graph-plotter', color='primary', style={'margin-bottom': '1em'}, block=True),
        # card place holder
        html.Div(id='data_desc', children=[]),
        # graph placeholder
        dcc.Graph(id='graph', figure={}),
        html.Br(),
        # dropdown selection #2
        dcc.Dropdown(
            id="graph_type2",
            options=[
                {'label': 'Tree Map', 'value': "tree"},
                {'label': 'Composite Plot', 'value': "comp"},
                {'label': 'Cloropleth Map', 'value': "map"}
            ],
            value='tree'
        ),
        # card placeholder #2
        html.Div(id='data_desc2', children=[]),
        # graph placeholder #2
        dcc.Graph(id='graph2', figure={}),
    ], style = {'backgroundColor': '#c1f0f0'})

    @app.callback(
        [Output(component_id='x_selection', component_property='options'),
         Output(component_id='y_selection', component_property='options'),
         Output(component_id='data_desc', component_property='children')],
        [Input(component_id='dataset_selection', component_property='value')]
    )
    def update_data(selected_data):
        # get the selected dataset
        dataset = datasets[selected_data]
        # set card description of count
        card = dbc.Card(
                        dbc.CardBody(
                        [html.H4(f"Row count of {selected_data} dataset is {len(dataset)}")]
                        )
        )
        # return the x and y selections as columns and the card
        return [{"label":c, "value":c} for c in dataset.columns], [{"label":c, "value":c} for c in dataset.columns], card

    @app.callback(
        [Output(component_id='graph', component_property='figure')],
        [Input('graph-plotter', 'n_clicks')],
        [State('x_selection', 'value'),
         State('y_selection', 'value'),
         State('dataset_selection', 'value'),
         State('graph_type', 'value')]
    )
    def update_graph(n_clicks, x_selection, y_selection, selected_data, graph_type):
        # get selected dataset
        dataset = datasets[selected_data]
        # get x and y columns
        x = dataset[x_selection]
        y = dataset[y_selection]
        fig = None
        # get selected graph and plot
        if graph_type == "bar":
            fig = px.bar(dataset, x=x_selection, y=y_selection)
        elif graph_type == "scatter":
            fig = px.scatter(dataset, x=x_selection, y=y_selection)
        elif graph_type == "line":
            fig = px.line(dataset, x=x_selection, y=y_selection)
        return fig,

    @app.callback(
        [Output(component_id='data_desc2', component_property='children'),
         Output(component_id='graph2', component_property='figure')],
        [Input(component_id='graph_type2', component_property='value')]
    )
    def update_graph2(graph_type):
        fig = None
        msg = None
        # get graph and plot w/ msg
        if graph_type == 'tree':
            fig = plotly_tree_map()
            msg = "Tree map of global expectancy life expectancy for 2000"
        elif graph_type == 'comp':
            fig = plotly_composite_line_bar()
            msg = "Composite plot for life expectancy dataset"
        elif graph_type == 'map':
            fig = plotly_map()
            msg = "Global Life Expectancy in 1920"
        # remove title from plots to replace with card titles
        fig.update_layout(
            title="")
        # set card title
        card = dbc.Card(
            dbc.CardBody(
                [html.H4(msg)]
            )
        )
        return card, fig

    return app




if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
   # app_ce = dash_callback_example()
    #app_b = dash_with_bootstrap_example()
    # app_c = dash_callback_example()
    app_t = dash_task()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # app_ce.run_server(debug=True)
    # app_b.run_server(debug=True)
    # app_c.run_server(debug=True)
    app_t.run_server(debug=True)
