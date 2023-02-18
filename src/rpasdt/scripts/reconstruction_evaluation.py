import csv

from rpasdt.algorithm.propagation_reconstruction import create_snapshot_IG, \
    _compute_neighbors_probability, _init_extended_network, \
    _check_if_node_is_on_path_between_infected_nodes
from rpasdt.scripts.sd_samples import get_experiments
from rpasdt.scripts.taxonomies import soc_anybeat, watts_strogatz_graph_2, \
    facebook, watts_strogatz_graph_1, barabasi_2, barabasi_1, footbal, dolphin, \
    karate_graph

DIR_NAME = "reconstruction"
graphs = [
    karate_graph,
    # dolphin,
    # footbal,
    # barabasi_1,
    # barabasi_2,
    # watts_strogatz_graph_1,
    # watts_strogatz_graph_2,
    # facebook,
    # soc_anybeat,
]
header = [
    "node",
    "neighbors_probability",
    "node_on_path",
    "infected",
]
deleted_ratios = [5, 10, 15, 20, 25, 30]


def _write_to_file(write_from_scratch, filename, data=None):
    print(f"Writing to {filename}")
    if write_from_scratch:
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()
    if not data:
        return
    file = open(filename, "a")
    csvwriter = csv.writer(file)
    csvwriter.writerow(data)
    file.close()


def generate_data(write_from_scratch=True):
    for graph_function in graphs:
        experiments = get_experiments(graph_function)

        for number_of_sources, experiments in experiments.items():

            for index, experiment in enumerate(experiments):
                for delete_ratio in deleted_ratios:
                    filename = f"results/{DIR_NAME}/{graph_function.__name__}_{number_of_sources}_{index}_{delete_ratio}.csv"
                    _write_to_file(write_from_scratch=True, filename=filename)

                    G = experiment.G
                    IG = experiment.IG
                    snapshot, _ = create_snapshot_IG(IG, delete_ratio)
                    EG = _init_extended_network(G=G, IG=snapshot)
                    for node in EG:
                        if node in snapshot:
                            continue
                        neighbors_probability = _compute_neighbors_probability(
                            node=node, G=EG)
                        node_on_path = int(
                            _check_if_node_is_on_path_between_infected_nodes(
                                node=node, G=EG)
                        )
                        _write_to_file(write_from_scratch=False, filename=filename,
                                       data=[node,neighbors_probability,node_on_path, node in IG])


generate_data()
