import matplotlib
import networkx as nx
from matplotlib import pyplot as plt
from utils import FIG_SIZE, draw_communities

from rpasdt.algorithm.df_community import blocd
from rpasdt.common.utils import get_object_value

PATH = "obrazki"
matplotlib.use("Qt5Agg")
plt.figure(figsize=FIG_SIZE)
# G = nx.krackhardt_kite_graph() teb dobry to zrozumienia community
G = nx.krackhardt_kite_graph()

# draw_communities(G, comm, "df_node_step1")


def draw_graph(G):
    pos = nx.spring_layout(G, seed=100)
    nx.draw_networkx(
        G,
    )


def draw_g():
    draw_graph(G)
    # save_plt(f"{PATH}/florentine.png")
    pass


def draw_1_step():
    comm = blocd(G, max_step=1)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step1")


def draw_2_step():
    comm = blocd(G, max_step=2)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step2")


def draw_3_step():
    comm = blocd(G, max_step=3)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step3")


draw_g()
draw_1_step()
draw_2_step()
draw_3_step()
