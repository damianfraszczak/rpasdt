import csv
import multiprocessing
import os
import time
from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any, Callable, Dict, List

import numpy as np
import stopit
from networkx import Graph

from rpasdt.algorithm.models import (
    PropagationReconstructionConfig,
    SingleSourceDetectionEvaluation,
    SourceDetectionSimulationConfig,
    SourceDetectionSimulationResult,
)
from rpasdt.algorithm.propagation_reconstruction import (
    NODE_INFECTION_PROBABILITY_ATTR,
    reconstruct_propagation,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.common.exceptions import log_error
from rpasdt.scripts.taxonomies import graphs, karate_graph, source_detectors
from rpasdt.scripts.utils import get_IG

THRESHOLDS = np.arange(0, 1.05, 0.1).round(2)

MINUTE = 60
TIMEOUT = 5 * MINUTE
WRITE_FROM_SCRATCH = False
DIR_NAME = "sd_samples"

header = [
    "type",
    "experiments",
    "sources",
    "detected",
    "threshold",
    "comm_difference",
    "avg_com_count",
    "avg_time",
    "avg_distance",
    "avg_detected_sources",
    "nr_of_empty_communities",
    "TP",
    "TN",
    "FP",
    "FN",
    "SUM",
    "TPR",
    "TNR",
    "PPV",
    "ACC",
    "F1",
]

header2 = [
    "type",
    "experiment",
    "sources",
    "detected",
    "normalized",
    "per_community",
    "infection_probability",
    "time",
    "nr_of_empty_communities",
    "error_distance",
    "TP",
    "TN",
    "FP",
    "FN",
    "SUM",
    "TPR",
    "TNR",
    "PPV",
    "ACC",
    "F1",
]
header_reconstruction = [
    "type",
    "experiment",
    "sources",
    "detected",
    "normalized",
    "per_community",
    "reconstructed_probability",
    "time",
    "nr_of_empty_communities",
    "error_distance",
    "TP",
    "TN",
    "FP",
    "FN",
    "SUM",
    "TPR",
    "TNR",
    "PPV",
    "ACC",
    "F1",
]


@dataclass
class Experiment:
    G: Graph
    IG: Graph
    graph_function: Any
    sources: List[int]


def get_experiments(graph_function) -> Dict[int, List[Experiment]]:
    G = graph_function()
    result = defaultdict(list)

    source_file = f"results/propagations/{graph_function.__name__}.csv"
    with open(source_file) as csvfile:
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            sources, infected_nodes = (
                row["sources"],
                row["infected_nodes"],
            )
            infected_nodes = infected_nodes.split("|")
            sources = sources.split("|")
            number_of_sources = len(sources)
            if number_of_sources == 13:
                continue

            IG = G.subgraph(infected_nodes)
            if len(IG.nodes) == 0:
                infected_nodes = [int(x) for x in infected_nodes]
                sources = [int(x) for x in sources]
                IG = G.subgraph(infected_nodes)
            result[number_of_sources].append(
                Experiment(G=G, IG=IG, graph_function=graph_function, sources=sources)
            )

    return result


def sd_work(ttt):
    with stopit.ThreadingTimeout(MINUTE * 10):
        name, experiment, source_detector, threshold = ttt[0], ttt[1], ttt[2], ttt[3]
        source_detector.config.source_threshold = threshold
        print(f"Processing {name}")
        start = time.time()
        sd_evaluation = source_detector.evaluate_sources(experiment.sources)
        end = time.time()
        sd_evaluation.additional_data["time"] = end - start
        sd_evaluation.additional_data["experiment"] = experiment
        return sd_evaluation


def do_evaluation():
    for graph_function in graphs:
        experiments = get_experiments(graph_function)
        filename = (
            f"results/{DIR_NAME}/{graph_function.__name__}_ce_static_network2222.csv"
        )
        if WRITE_FROM_SCRATCH:
            file = open(filename, "w")
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
            file.close()
        for number_of_sources, experiments in experiments.items():

            print(f"######## number_of_sources {number_of_sources}")
            aggregated_results = {}
            for experiment in experiments:
                sd_configs = {
                    name: config(None) for name, config in source_detectors.items()
                }
                sd_local = {
                    name: get_source_detector(
                        algorithm=config.alg,
                        G=experiment.G,
                        IG=experiment.IG,
                        config=config.config,
                        # number_of_sources=number_of_sources,
                    )
                    for name, config in sd_configs.items()
                }
                sd_local_items = list(sd_local.items())
                for threshold in THRESHOLDS:

                    result = aggregated_results.get(
                        threshold
                    ) or SourceDetectionSimulationResult(
                        source_detection_config=SourceDetectionSimulationConfig()
                    )
                    aggregated_results[threshold] = result
                    pool_obj = multiprocessing.Pool()
                    evaluations = pool_obj.map(
                        sd_work,
                        [
                            (name, experiment, source_detector, threshold)
                            for name, source_detector in sd_local.items()
                        ],
                    )

                    for i, evaluation in enumerate(evaluations):

                        name = sd_local_items[i][0]
                        source_detector = sd_local_items[i][1]
                        if evaluation is None:
                            print(f"NONE EVALUATION - {name}-{str(source_detector)}")
                            continue
                        result.add_result(
                            name, copy(source_detector.config), evaluation
                        )
                    pool_obj.close()

            # wszystkie thresholdy
            for threshold, aggregated_result in aggregated_results.items():
                for config, rr in aggregated_result.aggregated_results.items():
                    comm_difference, avg_com_count, nr_of_missing_communities = (
                        0.0,
                        0.0,
                        0.0,
                    )
                    for data in rr.additional_data:

                        communities = data["communities"]
                        experiment = data["experiment"]
                        detected_communities = len(communities.keys())
                        comm_difference += abs(detected_communities - number_of_sources)
                        avg_com_count += detected_communities
                        for cluster, nodes in communities.items():
                            if not any(s in nodes for s in experiment.sources):
                                nr_of_missing_communities += 1

                    detected_sources_sum = sum([len(s) for s in rr.detected_sources])

                    avg_com_count /= 1.0 * len(rr.additional_data)

                    row = [
                        config,
                        len(rr.additional_data),
                        number_of_sources,
                        threshold,
                        comm_difference,
                        avg_com_count,
                        rr.avg_execution_time,
                        rr.avg_error_distance,
                        detected_sources_sum * 1.0 / len(rr.detected_sources),
                        nr_of_missing_communities,
                        rr.TP,
                        rr.TN,
                        rr.FP,
                        rr.FN,
                        rr.P + rr.N,
                        rr.TPR,
                        rr.TNR,
                        rr.PPV,
                        rr.ACC,
                        rr.F1,
                    ]
                    file = open(filename, "a")
                    csvwriter = csv.writer(file)
                    csvwriter.writerow(row)
                    file.close()


def sd_evaluation_with_static_propagations():
    dir_name = "final_sd_results"
    for graph_function in graphs:
        G = graph_function()

        filename = f"results/{dir_name}/{graph_function.__name__}.csv"
        if WRITE_FROM_SCRATCH:
            file = open(filename, "w")
            csvwriter = csv.writer(file)
            csvwriter.writerow(header)
            file.close()
        print(f"############### {graph_function.__name__}")

        source_file = f"results/propagations/{graph_function.__name__}.csv"
        for threshold in THRESHOLDS:
            with open(source_file) as csvfile:
                spamreader = csv.DictReader(csvfile)
                aggregated_result = {}
                for row in spamreader:
                    sources, infected_nodes = (
                        row["sources"],
                        row["infected_nodes"],
                    )
                    IG, infected_nodes, sources = get_IG(G, infected_nodes, sources)

                    result = aggregated_result.get(
                        number_of_sources
                    ) or SourceDetectionSimulationResult(
                        source_detection_config=SourceDetectionSimulationConfig()
                    )
                    aggregated_result[number_of_sources] = result

                    sd_local = {
                        name: config(None) for name, config in source_detectors.items()
                    }

                    for (
                        name,
                        source_detector_config,
                    ) in sd_local.items():
                        print(f"PROCESSING {name}")
                        source_detector_config.config.source_threshold = threshold
                        source_detector = get_source_detector(
                            algorithm=source_detector_config.alg,
                            G=G,
                            IG=IG,
                            config=source_detector_config.config,
                            # number_of_sources=number_of_sources,
                        )

                        try:
                            with stopit.ThreadingTimeout(60):
                                start = time.time()
                                sd_evaluation = source_detector.evaluate_sources(
                                    sources
                                )
                                end = time.time()
                                sd_evaluation.additional_data["time"] = end - start
                                sd_evaluation.additional_data[
                                    "experiment"
                                ] = Experiment(
                                    sources=sources,
                                    IG=IG,
                                    G=G,
                                    graph_function=graph_function,
                                )
                                result.add_result(
                                    name, source_detector_config, sd_evaluation
                                )
                        except Exception as e:
                            log_error(exc=e, show_error_dialog=False)

                for number_of_sources, aggregated_result in aggregated_result.items():
                    for config, rr in aggregated_result.aggregated_results.items():
                        comm_difference, avg_com_count, nr_of_missing_communities = (
                            0.0,
                            0.0,
                            0.0,
                        )
                        for data in rr.additional_data:

                            communities = data["communities"]
                            experiment = data["experiment"]
                            detected_communities = len(communities.keys())
                            comm_difference += abs(
                                detected_communities - number_of_sources
                            )
                            avg_com_count += detected_communities
                            for cluster, nodes in communities.items():
                                if not any(s in nodes for s in experiment.sources):
                                    nr_of_missing_communities += 1

                        detected_sources_sum = sum(
                            [len(s) for s in rr.detected_sources]
                        )

                        avg_com_count /= 1.0 * len(rr.additional_data)

                        row = [
                            config,
                            len(rr.additional_data),
                            number_of_sources,
                            threshold,
                            comm_difference,
                            avg_com_count,
                            rr.avg_execution_time,
                            rr.error_distance,
                            detected_sources_sum * 1.0 / len(rr.detected_sources),
                            nr_of_missing_communities,
                            rr.TP,
                            rr.TN,
                            rr.FP,
                            rr.FN,
                            rr.P + rr.N,
                            rr.TPR,
                            rr.TNR,
                            rr.PPV,
                            rr.ACC,
                            rr.F1,
                        ]
                        file = open(filename, "a")
                        csvwriter = csv.writer(file)
                        csvwriter.writerow(row)
                        file.close()


def array_to_str(array):
    return ",".join([str(x) for x in array])


def is_in_file(filename, line):
    with open(filename) as f:
        for l in f:
            if line == l:
                return True
    return False


@dataclass
class DataToProcess:
    index: int
    key: str
    row: List
    graph_function: Callable
    G: Graph
    IG: Graph
    sources: List
    nodes_infection_probability: Dict = None
    file_postfix: str = ""
    results_dir: str = "final_sd_results"


def process_experiment(
    dp: DataToProcess,
):
    splitted_networks = ["facebook", "soc_anybeat"]
    key = dp.key
    graph_function = dp.graph_function
    G = dp.G
    IG, sources = dp.IG, dp.sources

    print(f"PROCESSING {key}-{graph_function.__name__}")

    dir_name = dp.results_dir
    network_file_name = graph_function.__name__
    if network_file_name in splitted_networks:
        network_file_name = f"{network_file_name}_{key}"
    filename = f"results/{dir_name}/{network_file_name}{dp.file_postfix}.csv"
    executed_experiments = set()
    if WRITE_FROM_SCRATCH or not os.path.exists(filename):
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header2)
        file.close()
    else:
        with open(filename) as f:
            for line in f:
                splitted = line.split(",")
                executed_experiments.add(f"{splitted[0]}-{splitted[1]}")
    sd_local = {name: config(None) for name, config in source_detectors.items()}

    for (
        name,
        source_detector_config,
    ) in sd_local.items():
        code = f"{name}-{key}"
        print(f"PROCESSING {code}")
        if code in executed_experiments:
            print(f"skipping {code}")
            continue

        try:
            source_detector = get_source_detector(
                algorithm=source_detector_config.alg,
                G=G,
                IG=IG,
                config=source_detector_config.config,
                # number_of_sources=number_of_sources,
            )

            start = time.time()
            sd_evaluation: SingleSourceDetectionEvaluation = (
                source_detector.evaluate_sources(sources)
            )
            end = time.time()
            final_time = end - start
            communities = sd_evaluation.additional_data["communities"]
            experiment = Experiment(
                sources=sources,
                IG=IG,
                G=G,
                graph_function=graph_function,
            )
            nr_of_missing_communities = 0
            for cluster, nodes in communities.items():
                if not any(s in nodes for s in experiment.sources):
                    nr_of_missing_communities += 1

            normalized_node_estimations = sd_evaluation.additional_data.get(
                "normalized_node_estimations", {}
            )
            estimations_per_community = sd_evaluation.additional_data.get(
                "estimation_per_community", {}
            )
            row = [
                name,
                key,
                array_to_str(sorted(sources)),
                array_to_str(sorted(sd_evaluation.detected_sources)),
                normalized_node_estimations,
                estimations_per_community,
                dp.nodes_infection_probability or {},
                final_time,
                nr_of_missing_communities,
                sd_evaluation.error_distance,
                sd_evaluation.TP,
                sd_evaluation.TN,
                sd_evaluation.FP,
                sd_evaluation.FN,
                sd_evaluation.P + sd_evaluation.N,
                sd_evaluation.TPR,
                sd_evaluation.TNR,
                sd_evaluation.PPV,
                sd_evaluation.ACC,
                sd_evaluation.F1,
            ]
            file = open(filename, "a")
            csvwriter = csv.writer(file)
            csvwriter.writerow(row)
            file.close()
        except Exception as e:
            print(f"ERROR {e}")


BEST_THRESHOLD = 0.439


# -0,29 88.74 -32.80
def read_reconstucted_ig(graph_name: str):
    filename = f"results/reconstruction/{graph_name}_evaluation.csv"
    ratio = 0.1
    result = defaultdict(dict)
    with open(filename) as csvfile:
        # index,number_of_sources,delete_ratio,deleted_nodes,reconstructed_nodes,m1,m2,threshold,ALL,TP,TN,FP,FN,accuracy,precision,recall,f1
        spamreader = csv.DictReader(csvfile)
        for row in spamreader:
            index = int(row["index"])
            threshold = int(float(row["threshold"]))
            if index != 0:
                continue
            if threshold != 1:
                continue
            delete_ratio = int(row["delete_ratio"])
            number_of_sources = int(row["number_of_sources"])
            key = f"{delete_ratio}-{number_of_sources}"

            if key in result:
                continue
            deleted_nodes = row["deleted_nodes"]
            # snapshot = IG.copy()
            # snapshot.remove_nodes_from([int(x) for x in deleted_nodes.split("|")])
            # newIg = reconstruct_propagation(
            #     PropagationReconstructionConfig(
            #         G=G,
            #         IG=snapshot,
            #         real_IG=IG,
            #         max_iterations=1,
            #     )
            # )
            result[key] = deleted_nodes
    return result


def sd_evaluation_with_reconstruction_final():
    for graph_function in graphs:
        print(f"############### {graph_function.__name__}")
        reconstructed_map = read_reconstucted_ig(graph_function.__name__)
        source_file = f"results/propagations/{graph_function.__name__}.csv"
        with open(source_file) as csvfile:
            with Pool(processes=4) as pool:
                spamreader = csv.DictReader(csvfile)
                for index, row in enumerate(spamreader):
                    if index % 3 != 0:
                        continue
                    G = graph_function()
                    sources, infected_nodes = (
                        row["sources"],
                        row["infected_nodes"],
                    )
                    IG, infected_nodes, sources = get_IG(G, infected_nodes, sources)
                    for ratio_sources, deleted_nodes in reconstructed_map.items():
                        ratio, number_of_sources = (
                            int(ratio_sources.split("-")[0]),
                            int(ratio_sources.split("-")[1]),
                        )
                        if number_of_sources != len(sources):
                            continue
                        print(
                            f"PROCESSING {graph_function.__name__}-{index}-{ratio}-{number_of_sources}"
                        )
                        snapshot = IG.copy()
                        snapshot.remove_nodes_from(
                            [int(x) for x in deleted_nodes.split("|")]
                        )
                        snapshot.remove_nodes_from(deleted_nodes.split("|"))
                        if len(snapshot) == len(IG):
                            raise Exception(
                                f"original {len(IG)} vs removed {len(snapshot)}"
                            )
                        EIG = reconstruct_propagation(
                            PropagationReconstructionConfig(
                                G=G,
                                IG=snapshot,
                                real_IG=IG,
                                max_iterations=1,
                            )
                        )
                        nodes_infection_probability = {
                            node[0]: node[1][NODE_INFECTION_PROBABILITY_ATTR]
                            for node in EIG.nodes(data=True)
                        }

                        data = [
                            DataToProcess(
                                index=index,
                                row=row,
                                graph_function=graph_function,
                                G=G,
                                IG=EIG,
                                sources=sources,
                                file_postfix="_reconstruction",
                                nodes_infection_probability=nodes_infection_probability,
                                results_dir="sd_with_reconstruction",
                                key=f"{index}-{ratio}",
                            )
                        ]
                        pool.map(process_experiment, data)


def sd_evaluation_final():
    for graph_function in graphs:
        print(f"############### {graph_function.__name__}")
        source_file = f"results/propagations/{graph_function.__name__}.csv"
        with open(source_file) as csvfile:
            with Pool(processes=4) as pool:
                spamreader = csv.DictReader(csvfile)
                for index, row in enumerate(spamreader):
                    G = graph_function()
                    sources, infected_nodes = (
                        row["sources"],
                        row["infected_nodes"],
                    )
                    IG, infected_nodes, sources = get_IG(G, infected_nodes, sources)
                    data = [
                        DataToProcess(
                            index=index,
                            row=row,
                            graph_function=graph_function,
                            G=G,
                            IG=IG,
                            sources=sources,
                            key=str(index),
                        )
                    ]
                    pool.map(process_experiment, data)


# do_evaluation()
# sd_evaluation_with_static_propagations()
if __name__ == "__main__":
    # sd_evaluation_final()
    sd_evaluation_with_reconstruction_final()
    # print(read_reconstucted_ig("karate_graph", karate_graph(), karate_graph()))
