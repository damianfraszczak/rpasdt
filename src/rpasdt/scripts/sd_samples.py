import csv
import time

import numpy as np
import stopit

from rpasdt.algorithm.models import (
    SourceDetectionSimulationConfig,
    SourceDetectionSimulationResult,
)
from rpasdt.algorithm.source_detectors.source_detection import (
    get_source_detector,
)
from rpasdt.common.exceptions import log_error
from rpasdt.scripts.taxonomies import graphs, source_detectors

THRESHOLDS = np.arange(0, 1, 0.5)

WRITE_FROM_BEGGINING = True


def sd_evaluation_with_static_propagations():
    header = [
        "type",
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

    for graph_function in graphs:
        G = graph_function()

        filename = f"results/sd_samples/{graph_function.__name__}_ce_static_network.csv"
        if WRITE_FROM_BEGGINING:
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
                        nr_of_missing_communities /= 1.0 * len(rr.additional_data)
                        row = [
                            config,
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


sd_evaluation_with_static_propagations()
