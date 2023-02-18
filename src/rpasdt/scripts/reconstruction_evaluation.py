import csv

from rpasdt.algorithm.propagation_reconstruction import (
    _check_if_node_is_on_path_between_infected_nodes,
    _compute_neighbors_probability,
    _init_extended_network,
    create_snapshot_IG,
)
from rpasdt.scripts.sd_samples import get_experiments
from rpasdt.scripts.taxonomies import (
    barabasi_1,
    barabasi_2,
    dolphin,
    facebook,
    footbal,
    karate_graph,
    soc_anybeat,
    watts_strogatz_graph_1,
    watts_strogatz_graph_2,
)

DIR_NAME = "reconstruction"
graphs = [
    karate_graph,
    dolphin,
    footbal,
    barabasi_1,
    barabasi_2,
    watts_strogatz_graph_1,
    watts_strogatz_graph_2,
    facebook,
    soc_anybeat,
]
header = [
    "number_of_sources",
    "index",
    "delete_ratio",
    "node",
    "neighbors_probability",
    "node_on_path",
    "infected",
    "sample_decision",
    "correct",
]
deleted_ratios = [5, 10, 15, 20, 25, 30]

grid_search = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]
threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]


def _write_to_file(filename, write_from_scratch=False, data=None):
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


def generate_data():
    for graph_function in graphs:
        experiments = get_experiments(graph_function)
        filename = f"results/{DIR_NAME}/{graph_function.__name__}.csv"
        _write_to_file(write_from_scratch=True, filename=filename)
        correct_count = 0
        incorrect_count = 0
        for number_of_sources, experiments in experiments.items():

            for index, experiment in enumerate(experiments):
                for delete_ratio in deleted_ratios:
                    G = experiment.G
                    IG = experiment.IG
                    snapshot, _ = create_snapshot_IG(IG, delete_ratio)
                    EG = _init_extended_network(G=G, IG=snapshot)
                    for node in EG:
                        if node in snapshot:
                            continue
                        neighbors_probability = _compute_neighbors_probability(
                            node=node, G=EG
                        )
                        node_on_path = int(
                            _check_if_node_is_on_path_between_infected_nodes(
                                node=node, G=EG
                            )
                        )
                        sample_result = 0.5 * neighbors_probability + 0.5 * node_on_path
                        decision = sample_result >= 0.8
                        correct = decision == (node in IG)
                        if correct:
                            correct_count += 1
                        else:
                            incorrect_count += 1
                        _write_to_file(
                            write_from_scratch=False,
                            filename=filename,
                            data=[
                                number_of_sources,
                                index,
                                delete_ratio,
                                node,
                                neighbors_probability,
                                node_on_path,
                                node in IG,
                                decision,
                                correct,
                            ],
                        )
        print(f"{filename}, correct: {correct_count}, incorrect: {incorrect_count}")


generate_data()
