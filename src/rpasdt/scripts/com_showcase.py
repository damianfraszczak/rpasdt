import networkx as nx
from constants import FIG_SIZE
from df_node_com import df_node_similarity
from matplotlib import pyplot as plt
from utils import draw_communities, draw_graph, get_object_value, save_plt

PATH = "obrazki"

plt.figure(figsize=FIG_SIZE)
# G = nx.krackhardt_kite_graph() teb dobry to zrozumienia community
G = nx.krackhardt_kite_graph()

# draw_communities(G, comm, "df_node_step1")


def draw_g():
    draw_graph(G)
    save_plt(f"{PATH}/florentine.png")


def draw_1_step():
    comm = df_node_similarity(G, max_step=1)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step1")


def draw_2_step():
    comm = df_node_similarity(G, max_step=2)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step2")


def draw_3_step():
    comm = df_node_similarity(G, max_step=3)
    comm = {
        index: community
        for index, community in enumerate(get_object_value(comm, "communities"))
    }
    draw_communities(G, comm, "df_node_step3")


# draw_g()
draw_1_step()
draw_2_step()
draw_3_step()
