import csv
import time

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

THRESHOLDS = [None, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def sd_evaluation_with_static_propagations():
    header = [
        "type",
        "sources",
        "threshold",
        "comm_difference",
        "avg_com_size",
        "avg_time",
        "avg_distance",
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
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()
        print(f"Proccesing {graph_function.__name__}")

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
                        name: config(number_of_sources)
                        for name, config in source_detectors.items()
                    }

                    for (
                        name,
                        source_detector_config,
                    ) in sd_local.items():
                        source_detector_config.config.source_threshold = threshold
                        source_detector = get_source_detector(
                            algorithm=source_detector_config.alg,
                            G=G,
                            IG=IG,
                            config=source_detector_config.config,
                            number_of_sources=number_of_sources,
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
                        comm_difference, avg_com_size = 0, 0
                        for data in rr.additional_data:
                            communities = data["communities"]
                            detected_communities = len(communities.keys())
                            comm_difference += abs(
                                detected_communities - number_of_sources
                            )
                            avg_com_size += detected_communities

                        avg_com_size /= len(rr.additional_data)
                        row = [
                            config,
                            number_of_sources,
                            threshold,
                            comm_difference,
                            avg_com_size,
                            rr.avg_execution_time,
                            rr.avg_error_distance,
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
