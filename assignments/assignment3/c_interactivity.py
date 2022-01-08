from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.widgets import Button, Slider, RadioButtons

from assignments.assignment1.a_load_file import *
from assignments.assignment1.e_experimentation import *
from assignments.assignment3.b_simple_usages import *


###############
# Interactivity in visualizations is challenging due to limitations and chunkiness of libraries.
# For example, some libraries works well in Jupyter Notebooks, but then the code makes barely any sense and becomes hard to change/update,
# defeating the purpose of using Jupyter notebooks in the first place, and other libraries provide a window of their own, but
# they are very tied to the running code, and far from the experience you'd expect from a proper visual analytics webapp
#
# We will try out some approaches to exercise in this file, but the next file will give you the proper tooling to make a
# well-rounded and efficient code for visual interactions.
##############################################
# Example(s). Read the comments in the following method(s)
##############################################


def matplotlib_simple_example():
    """
    Using the same logic from before, we can add sliders or buttons to select options for interactivity.
    Matplotlib is limited to only a few options, but do take a look, since they are useful for fast prototyping and analysis

    In case you are using PyCharm, I suggest you to uncheck the 'Show plots in tool window'
    to be able to see and interact with the buttons defined below.
    This example comes from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html
    """
    freqs = np.arange(2, 20, 3)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    t = np.arange(0.0, 1.0, 0.001)
    s = np.sin(2 * np.pi * freqs[0] * t)
    l, = plt.plot(t, s, lw=2)

    class Index(object):
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

        def prev(self, event):
            print(event)
            self.ind -= 1
            i = self.ind % len(freqs)
            ydata = np.sin(2 * np.pi * freqs[i] * t)
            l.set_ydata(ydata)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)
    return fig, ax


def matplotlib_simple_example2():
    """
    Here is another example, which also has a slider and simplifies a bit the callbacks
    """
    data = np.random.rand(10, 5)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.bar(np.arange(10).astype(str).tolist(), data[:, 0])

    class Index(object):
        ind = 0
        multiplier = 1

        def change_data(self, event, i):
            self.ind = np.clip(self.ind + i, 0, data.shape[1] - 1)
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

        def change_multiplier(self, value):
            self.multiplier = value
            ax.clear()
            ax.bar(np.arange(10).astype(str).tolist(), data[:, self.ind] * self.multiplier)
            plt.draw()

    callback = Index()
    axprev = plt.axes([0.1, 0.05, 0.12, 0.075])
    axnext = plt.axes([0.23, 0.05, 0.12, 0.075])
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(lambda event: callback.change_data(event, 1))
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(lambda event: callback.change_data(event, -1))
    slider = Slider(axslider, 'multiplier', 1, 10, 1)
    slider.on_changed(callback.change_multiplier)
    return fig, ax


def plotly_slider_example():
    """
    Here is a simple example from https://plotly.com/python/sliders/ of how to include a slider in plotly
    Notice how similar it already is to GapMinder!
    """
    df = px.data.gapminder()
    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                     animation_frame="year",  # set which column makes the animation though a slider
                     size="pop",
                     color="continent",
                     hover_name="country",
                     log_x=True,
                     size_max=55,
                     range_x=[100, 100000],
                     range_y=[25, 90])

    fig["layout"].pop("updatemenus")  # optional, drop animation buttons

    return fig


def plotly_button_example():
    """
    To have buttons, plotly requires us to use go (and not px) to generate the graph.
    The button options are very restrictive, since all they can do is change a parameter from the go graph.
    In the example below, it changes the "mode" value of the graph (between lines and scatter)
    The code is a modified code example taken from https://plotly.com/python/custom-buttons/
    """
    x = np.random.rand(50) * np.random.randint(-10, 10)
    y = np.random.rand(50) * np.random.randint(-10, 10)

    fig = go.Figure()

    # Add surface trace
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers'))

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            dict(type="buttons",
                 direction="left",
                 buttons=[
                     dict(
                         label="line",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "markers"}],  # This is the value being updated in the visualization
                     ), dict(
                         label="scatter",  # just the name of the button
                         method="update",  # This is the method of update (check https://plotly.com/python/custom-buttons/)
                         args=[{"mode": "line"}],  # This is the value being updated in the visualization
                     )
                 ],
                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"  # Layout-related values
                 ),
        ]
    )
    return fig


##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
##############################################
def matplotlib_interactivity():
    """
    Do an interactive matplotlib plot where I can select which visualization I want.
    Make either a slider, a dropdown or several buttons and make so each option gives me a different visualization from
    the matplotlib figures of b_simple_usages. Return just the resulting fig as is done in plotly_slider_example.
    """
    # note to use in a more 'functional' way, globals() is employed
    # default to bar chart
    global fig, ax
    fig, ax = matplotlib_bar_chart()
    plt.subplots_adjust(bottom=0.2)
    class Index(object):
        def __init__(self):
            # loading data on init, so we dont need to load every time we switch figures
            # bar
            self.i_data = read_dataset(Path('..', '..', 'iris.csv'))
            self.bar_data =  [get_column_max(self.i_data, col_name) for col_name in get_numeric_columns(self.i_data)]
            # pie
            data = process_life_expectancy_dataset()
            self.pie_data = [
                len(get_numeric_columns(data)),
                len(get_binary_columns(data)),
                len(get_text_categorical_columns(data))
            ]
            self.pie_labels = ["Numeric", "Binary", "Categorical"]
            # hist
            self.hist_data = get_numeric_columns(self.i_data)
            # heatmap
            self.heat_data = self.i_data[get_numeric_columns(self.i_data)].corr(method="pearson")
            self.num_axes = 1
        def bar(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].bar(range(len(self.bar_data)), height=self.bar_data)
            plt.draw()
        def pie(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].pie(self.pie_data, labels=self.pie_labels)
            plt.draw()

        def hist1(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].hist(self.i_data[self.hist_data[0]])
            plt.draw()

        def hist2(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].hist(self.i_data[self.hist_data[1]])
            plt.draw()
        def hist3(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].hist(self.i_data[self.hist_data[2]])
            plt.draw()
        def hist4(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].hist(self.i_data[self.hist_data[3]])
            plt.draw()
        def heat(self, event):
            # check to see if we need to change the ax numbers
            globals()["ax"].clear()
            globals()["ax"].imshow(self.heat_data, cmap='hot')
            plt.draw()
    # initialize the callback
    callback = Index()
    # set axes
    bar = plt.axes([0.1, 0.05, 0.1, 0.075])
    pie = plt.axes([0.2, 0.05, 0.1, 0.075])
    hist1 = plt.axes([0.3, 0.05, 0.1, 0.075])
    hist2 = plt.axes([0.4, 0.05, 0.1, 0.075])
    hist3 = plt.axes([0.5, 0.05, 0.1, 0.075])
    hist4 = plt.axes([0.6, 0.05, 0.1, 0.075])
    heat = plt.axes([0.7, 0.05, 0.1, 0.075])
    # Buttons #
    # bar
    bar = Button(bar, 'bar plot')
    bar.on_clicked(callback.bar)
    # pie
    pie = Button(pie, 'pie plot')
    pie.on_clicked(callback.pie)
    # histogram (split into four to prevent switching from 1x1 to 2x2 graph)
    hist1 = Button(hist1, 'hist1')
    hist2 = Button(hist2, 'hist2')
    hist3 = Button(hist3, 'hist3')
    hist4 = Button(hist4, 'hist4')
    hist1.on_clicked(callback.hist1)
    hist2.on_clicked(callback.hist2)
    hist3.on_clicked(callback.hist3)
    hist4.on_clicked(callback.hist4)
    heat = Button(heat, 'heat')
    heat.on_clicked(callback.heat)
    fig.suptitle("Interactive MPL plot")
    return fig



def matplotlib_cluster_interactivity():
    """
    Do an interactive matplotlib plot where I can select how many clusters I want to train from.
    Use iris dataset (just numeric columns) and k-means (feel free to reuse as/c_clustering if you wish).
    The slider (or dropdown) should range from 2 to 10. Return just the resulting fig.
    """

    # get iris data  (code snippet borrowed from c_clustering.py/iris_clusters)
    iris = process_iris_dataset_again()
    ohe = generate_one_hot_encoder(iris['species'])
    data = replace_with_one_hot_encoder(iris.copy(deep=False), 'species', ohe, list(ohe.get_feature_names()))
    # intialize plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax.scatter(x=data["sepal_length"], y=data["sepal_width"])

    class Index(object):
        def __init__(self):
            self.k = 2

        def train_k(self, value):
            self.k = value
            ax.clear()
            result = simple_k_means_pred(x=data, n_clusters=self.k)
            ax.scatter(x=data["sepal_length"], y=data["sepal_width"], c=result["clusters"])
            plt.draw()
    # intialize callback
    callback = Index()
    # ax slider set to discrete values to set number of clusters
    axslider = plt.axes([0.55, 0.1, 0.35, 0.03])
    slider = Slider(axslider, 'K', 2, 10, valinit=2, valstep=1)
    slider.on_changed(callback.train_k)
    fig.suptitle("Interactive Iris Clustering")
    return fig


def plotly_interactivity():
    """
    Do a plotly graph with all plotly 6 figs from b_simple_usages, and make 6 buttons (one for each fig).
    Change the displayed graph depending on which button I click. Return just the resulting fig.
    """
    # note for the purpose of this question, it was more suitable to recreate the plots to add to traces #

    # DATA #

    # iris cluster data #
    clusters = cluster_iris_dataset_again()
    # read iris
    i_data = read_dataset(Path('..', '..', 'iris.csv'))
    # add cluster info to iris df
    i_data["clusters"] = clusters["clusters"]

    # polar plot data#
    # adjusted preprocessed data set that includes necessary data
    data = process_life_expectancy_dataset_lon()
    # store all results of conversions in df
    life_df = pd.DataFrame()
    # radius = sum of values for all years (normalized)
    life_df["r"] = np.array(data.groupby("country")["value"].sum())
    # x = latitude
    x = np.array(data.groupby("country")["Latitude"].unique().apply(lambda x: x[0]))
    # y = longitude
    y = np.array(data.groupby("country")["Longitude"].unique().apply(lambda x: x[0]))
    # compute theta same as previous examples (convert to degrees)
    life_df["theta"] = np.arctan(y / x) * 57.2958

    # composite plot data #
    data = process_life_expectancy_dataset_lon()
    # select 5 countries
    countries = ['Slovak Republic', 'Germany', 'Hungary', 'Czech Republic', 'Poland']
    composite_data = data[data["country"].isin(countries)]
    composite_sum_data = composite_data.groupby("year")["value"].sum()

    # tree map data #
    tree_data = process_life_expectancy_dataset_unnorm().query("year == 2000")
    tree_labels = list(tree_data['country']) + list(tree_data['continent'].unique()) + ["world"]
    tree_parents = list(tree_data['continent']) + ["world" for _ in range(len(list(tree_data['continent'].unique())))] + [""]
    tree_values = list(tree_data['value']) + [tree_data[tree_data['continent'] == cont]["value"].sum() for cont in
                                         tree_data['continent']] + [tree_data["value"].sum()]

    # IPLOT #
    # init figure
    fig = go.Figure()
    # approach taken: adjsut visibility for toggled plots #
    # create a mask to toggle visibility for different figures
    visibility_mask = []

    # plotly_bar_plot_chart()
    for cluster in i_data["clusters"].unique():
        fig.add_trace(go.Bar(
               x=list(dict(i_data[i_data["clusters"] == cluster]["species"].value_counts()).keys()),
               y=list(dict(i_data[i_data["clusters"] == cluster]["species"].value_counts()).values()),name=f"Cluster {cluster}", visible=True))
        visibility_mask.append("bar plot")

    # plotly_scatter_plot_chart()
    fig.add_trace(go.Scatter(x=i_data['sepal_length'],
                             y=i_data['sepal_width'],
                             marker=dict(color=i_data["clusters"]),
                             name='scatter',
                             mode='markers', visible=False))
    visibility_mask.append("scatter plot")

    # plotly_composite_line_bar
    fig.add_trace(go.Bar(x=composite_sum_data.index, y=np.array(composite_sum_data), name="sum", visible=False))
    visibility_mask.append("composite plot")
    for country in countries:
        country_data = data[data["country"] == country]
        fig.add_trace(go.Scatter(x=country_data["year"], y=country_data["value"], mode="lines", name=country, visible=False))
        visibility_mask.append("composite plot")

    # plotly_tree_map()

    fig.add_trace(go.Treemap(
                    labels = tree_labels,
                    parents = tree_parents,
                    values= tree_values,
                    root_color="lightgrey", visible=False
                ))
    visibility_mask.append("tree map")

    # Add dropdown for all associated plots
    fig.update_layout(barmode='group',
                      title='Interactive Plot',
                        updatemenus=[
                            dict(type="buttons",
                                 direction="left",
                                 buttons=[
                                     dict(
                                         label="scatter plot",  # just the name of the button
                                         method="update",
                                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                                         args=[{'visible': np.array(visibility_mask) == "scatter plot"}],  # This is the value being updated in the visualization
                                     ),
                                     dict(
                                         label="bar plot",
                                         method="update",
                                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                                         args=[{'visible': np.array(visibility_mask) == "bar plot"}],  # This is the value being updated in the visualization
                                     ),
                                     dict(
                                         label="composite plot",  # just the name of the button
                                         method="update",
                                         # This is the method of update (check https://plotly.com/python/custom-buttons/)
                                         args=[{'visible':  np.array(visibility_mask) == "composite plot"}]
                                     ),
                                     dict(
                                         label="tree map",  # just the name of the button
                                         method="update",
                                         args=[{'visible': np.array(visibility_mask) == "tree map"}]
                                     )
                                 ],
                                 pad={"r": 10, "t": 10}, showactive=True, x=0.11, xanchor="left", y=1.1, yanchor="top"
                                 ),
                        ]
                    )
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_i = matplotlib_interactivity()
    fig_m_ci = matplotlib_cluster_interactivity()
    # fig_p =  plotly_interactivity()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # matplotlib_simple_example()[0]
    # plt.show()
    # matplotlib_simple_example2()[0]
    # plt.show()
    # plotly_slider_example().show()
    # plotly_button_example().show()
    # fig_m_i.show()
    # fig_m_ci.show()
    # fig_p.show()
