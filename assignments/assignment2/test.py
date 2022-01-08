# import plotly.graph_objects as go
from assignments.assignment3.b_simple_usages import *
# # tree map data
tree_data = process_life_expectancy_dataset_unnorm().query("year == 2000")
#
#
# fig = go.Figure(go.Treemap(
#                      labels=tree_data['country'][:5],
#                      parents=list(tree_data['continent'][:5]),
#                      values=tree_data['value'][:5],
#                              visible=True
#                      # color=tree_data['value']
#                ))
# fig.show()
import plotly.graph_objects as go
labels = list(tree_data['country']) + list(tree_data['continent'].unique()) + ["world"]
parents = list(tree_data['continent']) + ["world" for _ in range(len(list(tree_data['continent'].unique())))] + [""]
values = list(tree_data['value']) +[tree_data[tree_data['continent'] == cont]["value"].sum() for cont in tree_data['continent']] +[tree_data["value"].sum()]
fig = go.Figure(go.Treemap(
    labels = labels,
    parents = parents,
    values= values,
    root_color="lightgrey"
))

fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()
print()