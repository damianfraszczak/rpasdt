import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import stopit
from networkx import Graph

from rpasdt.algorithm.models import (
    SourceDetectionSimulationConfig,
    SourceDetectionSimulationResult,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.common.exceptions import log_error
from rpasdt.scripts.taxonomies import graphs, source_detectors

THRESHOLDS = np.arange(0, 1, 0.1).round(2)

WRITE_FROM_SCRATCH = True
DIR_NAME = "sd_samples"

header = [
    "type",
    "experiments",
    "sources",
    "threshold",
    "comm_difference",
    "avg_com_count",
    "avg_time",
    "avg_distance",
    "avg_detected_sources",
    "nr_of_missing_communities",
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

            IG = G.subgraph(infected_nodes)
            if len(IG.nodes) == 0:
                infected_nodes = [int(x) for x in infected_nodes]
                sources = [int(x) for x in sources]
                IG = G.subgraph(infected_nodes)
            result[number_of_sources].append(
                Experiment(G=G, IG=IG, graph_function=graph_function, sources=sources)
            )

    return result


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
                sd_local = {
                    name: config(None) for name, config in source_detectors.items()
                }

                for threshold in THRESHOLDS:

                    result = aggregated_results.get(
                        threshold
                    ) or SourceDetectionSimulationResult(
                        source_detection_config=SourceDetectionSimulationConfig()
                    )
                    aggregated_results[threshold] = result

                    for (
                        name,
                        source_detector_config,
                    ) in sd_local.items():
                        print(f"PROCESSING {name}, threshold = {threshold}")
                        source_detector_config.config.source_threshold = threshold
                        source_detector = get_source_detector(
                            algorithm=source_detector_config.alg,
                            G=experiment.G,
                            IG=experiment.IG,
                            config=source_detector_config.config,
                            # number_of_sources=number_of_sources,
                        )

                        try:
                            with stopit.ThreadingTimeout(60):
                                start = time.time()
                                sd_evaluation = source_detector.evaluate_sources(
                                    experiment.sources
                                )
                                end = time.time()
                                sd_evaluation.additional_data["time"] = end - start
                                result.add_result(
                                    name, source_detector_config, sd_evaluation
                                )
                        except Exception as e:
                            log_error(exc=e, show_error_dialog=False)
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
    for graph_function in graphs:
        G = graph_function()

        filename = f"results/{DIR_NAME}/{graph_function.__name__}_ce_static_network.csv"
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
                    infected_nodes = infected_nodes.split("|")
                    sources = sources.split("|")
                    number_of_sources = len(sources)

                    IG = G.subgraph(infected_nodes)
                    if len(IG.nodes) == 0:
                        infected_nodes = [int(x) for x in infected_nodes]
                        sources = [int(x) for x in sources]
                        IG = G.subgraph(infected_nodes)

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
                            detected_communities = len(communities.keys())
                            comm_difference += abs(
                                detected_communities - number_of_sources
                            )
                            avg_com_count += detected_communities
                            for cluster, nodes in communities.items():
                                if not any(s in nodes for s in sources):
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


do_evaluation()
