from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from assignments.assignment1.b_data_profile import *
from assignments.assignment1.c_data_cleaning import *
from assignments.assignment1.d_data_encoding import *
from assignments.assignment1.a_load_file import *
from assignments.assignment1.e_experimentation import *

from assignments.assignment2.c_clustering import *
from assignments.assignment2.a_classification import *
##############################################
# In this file, we will use data and methods of previous assignments with visualization.
# But before you continue on, take some time to look on the internet about the many existing visualization types and their usages, for example:
# https://extremepresentation.typepad.com/blog/2006/09/choosing_a_good.html
# https://datavizcatalogue.com/
# https://plotly.com/python/
# https://www.tableau.com/learn/whitepapers/which-chart-or-graph-is-right-for-you
# Or just google "which visualization to use", and you'll find a near-infinite number of resources
#
# You may want to create a new visualization in the future, and for that I suggest using JavaScript and D3.js, but for the course, we will only
# use python and already available visualizations
##############################################
# Implement all the below methods
# Use the documentation of the libraries in case you need help to understand the required methods
# For ALL methods return the fig and ax of matplotlib or fig from plotly!
##############################################
def matplotlib_bar_chart() -> Tuple:
    """
    Create a bar chart with a1/b_data_profile's get column max.
    Show the max of each numeric column from iris dataset as the bars
    """
    # reading straight from iris datset
    data = read_dataset(Path('..', '..', 'iris.csv'))
    # get max for each column
    x = [get_column_max(data, col_name) for col_name in get_numeric_columns(data)]
    # plot
    fig, ax = plt.subplots()
    ax.bar(range(len(x)), height=x)
    fig.suptitle('Iris column max')
    ax.set_xlabel('column')
    ax.set_ylabel('count')
    return fig, ax


def matplotlib_pie_chart() -> Tuple:
    """
    Create a pie chart where each piece of the chart has the number of columns which are numeric/categorical/binary
    from the output of a1/e_/process_life_expectancy_dataset
    """
    data = process_life_expectancy_dataset()
    pie_data = [
            len(get_numeric_columns(data)),
            len(get_binary_columns(data)),
            len(get_text_categorical_columns(data))
    ]
    labels = ["Numeric", "Binary", "Categorical"]
    fig, ax = plt.subplots()
    ax.pie(pie_data, labels=labels)
    fig.suptitle('Life Expectency Data Types')
    return fig, ax


def matplotlib_histogram() -> Tuple:
    """
    Build 4 histograms as subplots in one figure with the numeric values of the iris dataset
    """
    data = read_dataset(Path('..', '..', 'iris.csv'))
    cols = get_numeric_columns(data)

    fig, ax = plt.subplots(2,2)
    ax[0,0].hist(data[cols[0]])
    ax[0,1].hist(data[cols[1]])
    ax[1,0].hist(data[cols[2]])
    ax[1,1].hist(data[cols[3]])
    # set individual titles
    ax[0, 0].set_title(cols[0])
    ax[0, 1].set_title(cols[1])
    ax[1, 0].set_title(cols[2])
    ax[1, 1].set_title(cols[3])
    # ax abels
    fig.supxlabel('bins')
    fig.supylabel('counts')
    return fig, ax



def matplotlib_heatmap_chart() -> Tuple:
    """
    Remember a1/b_/pandas_profile? There is a heat map over there to analyse the correlation among columns.
    Use the pearson correlation (e.g. https://docs.scipy.org/doc/scipy-1.5.3/reference/generated/scipy.stats.pearsonr.html)
    to calculate the correlation between two numeric columns and show that as a heat map. Use the iris dataset.
    """
    # read original iris dataset
    data = read_dataset(Path('..', '..', 'iris.csv'))
    data = data[get_numeric_columns(data)]
    # compute pearson correlation
    data = data.corr(method="pearson")
    # plot
    fig, ax = plt.subplots()
    ax.imshow(data, cmap='hot')
    fig.suptitle('Correlation of Iris columns')
    return fig, ax


# There are many other possibilities. Please, do check the documentation and examples so you
# may have a good breadth of tools for future work (in assignments, projects, and your own career)
###################################
# Once again, for ALL methods return the fig and ax of matplotlib or fig from plotly!


def plotly_scatter_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() as the color of a scatterplot made from the original (unprocessed)
    iris dataset. Choose among the numeric values to be the x and y coordinates.
    """
    # get cluster information
    clusters = cluster_iris_dataset_again()
    # read iris
    data = read_dataset(Path('..', '..', 'iris.csv'))
    # add cluster info to iris df
    data["clusters"] = clusters["clusters"]
    fig = px.scatter(data, x='sepal_length', y='sepal_width', color="clusters")
    fig.update_layout(
        title="Iris Clustering Result")
    return fig


def plotly_bar_plot_chart():
    """
    Use the result of a2/c_clustering/cluster_iris_dataset_again() and use x as 3 groups of bars (one for each iris species)
    and each group has multiple bars, one for each cluster, with y as the count of instances in the specific cluster/species combination.
    The grouped bar chart is like https://plotly.com/python/bar-charts/#grouped-bar-chart (search for the grouped bar chart visualization)
    """
    # get cluster information
    clusters = cluster_iris_dataset_again()
    # read iris
    data = read_dataset(Path('..', '..', 'iris.csv'))
    # add cluster info to iris df
    data["clusters"] = clusters["clusters"]

    fig = go.Figure(data=[
                        # make barchart that shows the different clusters wrt different species
                        go.Bar(name=f"Cluster {cluster}",
                               x=list(dict(data[data["clusters"] == cluster]["species"].value_counts()).keys()),
                               y=list(dict(data[data["clusters"] == cluster]["species"].value_counts()).values()))
                               for cluster in data["clusters"].unique()
                          ])
    # Change the bar mode and set title
    fig.update_layout(barmode='group',
                      title="Grouped Bars for Iris Clusters",
                      xaxis_title="Clusters Grouped by Species",
                      yaxis_title="Count"
                      )
    return fig


def plotly_polar_scatterplot_chart():
    """
    Do something similar to a1/e_/process_life_expectancy_dataset, but don't drop the latitude and longitude.
    Use these two values to figure out the theta to plot values as a compass (example: https://plotly.com/python/polar-chart/).
    Each point should be one country and the radius should be thd value from the dataset (add up all years and feel free to ignore everything else)
    """
    """
    Using an azimuthal equidistant projection https://en.wikipedia.org/wiki/Azimuthal_equidistant_projection
    and considering the 
    """
    # adjusted preprocessed data set that includes necessary data
    data = process_life_expectancy_dataset_lon()
    # store all results of conversions in df
    df = pd.DataFrame()
    # radius = sum of values for all years (normalized)
    df["r"] = np.array(data.groupby("country")["value"].sum())
    # x = latitude
    x = np.array(data.groupby("country")["Latitude"].unique().apply(lambda x: x[0]))
    # y = longitude
    y = np.array(data.groupby("country")["Longitude"].unique().apply(lambda x: x[0]))
    # compute theta same as previous examples (convert to degrees)
    df["theta"] = np.arctan(y / x) * 57.2958
    fig = px.scatter_polar(df, r="r", theta="theta")
    fig.update_layout(title="Polar plot for Life Expectency dataset")
    return fig


def plotly_table():
    """
    Show the data from a2/a_classification/decision_tree_classifier() as a table
    See https://plotly.com/python/table/ for documentation
    """
    # load data
    iris = read_dataset(Path('../../iris.csv'))
    label_col = "species"
    feature_cols = iris.columns.tolist()
    feature_cols.remove(label_col)
    x_iris = iris[feature_cols]
    y_iris = iris[label_col]
    # run classifier
    X_test, y_test, y_pred = decision_tree_classifier_a3(x_iris, y_iris)
    # store in results in df
    data = pd.DataFrame(X_test, columns=["sepal_length" , "sepal_width", "petal_length", "petal_width"])
    data["species"] = y_test
    data["dt_prediction"] = y_pred
    # make table
    fig = go.Figure(data=[go.Table(header=dict(values=list(data.columns)),
                                   cells=dict(values=[list(data[col]) for col in data.columns]))
                          ])
    fig.update_layout(title="Decision Tree Classifier for Iris dataset")
    return fig


def plotly_composite_line_bar():
    """
    Use the data from a1/e_/process_life_expectancy_dataset and show in a single graph on year on x and value on y where
    there are 5 line charts of 5 countries (you choose which) and one bar chart on the background with the total value of all 5
    countries added up.
    """
    data = process_life_expectancy_dataset_lon()
    # select 5 countries
    countries = ['Slovak Republic', 'Germany', 'Hungary', 'Czech Republic','Poland']
    data = data[data["country"].isin(countries)]
    sum_data = data.groupby("year")["value"].sum()
    # plot
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sum_data.index, y=np.array(sum_data), name="sum"))
    # add each country individually
    for country in countries:
        country_data = data[data["country"] == country]
        fig.add_trace(go.Scatter(x=country_data["year"], y=country_data["value"], mode="lines", name=country))
    # update title and labels
    fig.update_layout(barmode='stack', title=f"Life Expectancy vs. Year for {', '.join(countries[:-1])} and {countries[-1]}")
    fig.update_xaxes(title="Life Expectancy")
    fig.update_yaxes(title="Year")
    return fig


def plotly_map():
    """
    Use the data from a1/e_/process_life_expectancy_dataset on a plotly map (anyone will do)
    Examples: https://plotly.com/python/maps/, https://plotly.com/python/choropleth-maps/#using-builtin-country-and-state-geometries
    Use the value from the dataset of a specific year (e.g. 1900) to show as the color in the map
    """
    # adjusted preprocessed data set that includes necessary data
    data = process_life_expectancy_dataset_lon()
    # plot the life expectency for 1920
    fig = px.choropleth(data[data["year"] == 1920], locations="country",locationmode="country names",
                        color="value",
                        hover_name="country",
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig.update_layout(title="Global Life Expectancy in 1920")
    return fig


def plotly_tree_map():
    """
    Use plotly's treemap to plot any data returned from any of a1/e_experimentation or a2 tasks
    Documentation: https://plotly.com/python/treemaps/
    """
    # load unnormalized dataset
    data = process_life_expectancy_dataset_unnorm().query("year == 2000")
    fig = px.treemap(data, path=[px.Constant("world"), 'continent', 'country'],
                     values='value',
                     color='value',
                     hover_data=['country'],
                    )
    fig.update_layout(title="Tree map of global expectancy life expectancy for 2000")
    return fig


if __name__ == "__main__":
    # Here are examples of how to run each method
    # Although you may remove or comment them out for testing, make sure they are present
    # and work when submitting, since when we run your code, all methods will be run like this.
    # If these lines below returns errors when running, your file will be considered to not
    # run, and graded accordingly.
    fig_m_bc, _ = matplotlib_bar_chart()
    fig_m_pc, _ = matplotlib_pie_chart()
    fig_m_h, _ = matplotlib_histogram()
    fig_m_hc, _ = matplotlib_heatmap_chart()

    fig_p_s = plotly_scatter_plot_chart()
    fig_p_bpc = plotly_bar_plot_chart()
    fig_p_psc = plotly_polar_scatterplot_chart()
    fig_p_t = plotly_table()
    fig_p_clb = plotly_composite_line_bar()
    fig_p_map = plotly_map()
    fig_p_treemap = plotly_tree_map()

    # Uncomment the below lines to test your code
    # When submitting, leave the code below commented!!!
    # fig_m_bc.show()
    # fig_m_pc.show()
    # fig_m_h.show()
    # fig_m_hc.show()
    #
    # fig_p_s.show()
    # fig_p_bpc.show()
    # fig_p_psc.show()
    # fig_p_t.show()
    # fig_p_clb.show()
    # fig_p_map.show()
    # fig_p_treemap.show()
