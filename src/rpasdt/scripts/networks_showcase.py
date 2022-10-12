import networkx as nx

from rpasdt.scripts.taxonomies import buzznet

# graph = nx.karate_club_graph()
# print(get_power_law(graph))
#

graph = buzznet()
print(nx.average_shortest_path_length(graph))
