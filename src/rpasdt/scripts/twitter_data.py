import os

import matplotlib

# import matplotlib
import networkx as nx
from matplotlib import pyplot as plt

#
# from rpasdt.common.utils import get_project_root
matplotlib.use("Qt5Agg")
#
# def data_ego_553587013409325058():
#     return nx.read_adjlist(
#         os.path.join(get_project_root(), "data", "twitter",
#                      "ego-graph-553587013409325058.adjlist"))


G = nx.complete_graph(5)
nx.draw_networkx(G)
plt.show()
