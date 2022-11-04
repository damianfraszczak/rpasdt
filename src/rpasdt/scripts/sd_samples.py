import csv

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


def sd_evaluation_with_static_propagations():
    header = [
        "type",
        "sources",
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
    aggregated_result = {}
    for graph_function in graphs:
        G = graph_function()

        filename = f"results/sd_samples/{graph_function.__name__}_ce_static_network.csv"
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()
        print(f"Proccesing {graph_function.__name__}")

        source_file = f"results/propagations/{graph_function.__name__}.csv"
        with open(source_file) as csvfile:
            spamreader = csv.DictReader(csvfile)
            for row in spamreader:
                sources, infected_nodes = (
                    row["sources"],
                    row["infected_nodes"],
                )
                infected_nodes = [int(x) for x in infected_nodes.split("|")]
                sources = [int(x) for x in sources.split("|")]
                number_of_sources = len(sources)
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
                    source_detector = get_source_detector(
                        algorithm=source_detector_config.alg,
                        G=G,
                        IG=IG,
                        config=source_detector_config.config,
                        number_of_sources=number_of_sources,
                    )
                    try:
                        with stopit.ThreadingTimeout(1000):
                            sd_evaluation = source_detector.evaluate_sources(sources)
                            result.add_result(
                                name, source_detector_config, sd_evaluation
                            )
                    except Exception as e:
                        log_error(exc=e, show_error_dialog=False)

        for number_of_sources, aggregated_result in aggregated_result.items():
            for config, rr in aggregated_result.aggregated_results.items():
                row = [
                    config,
                    number_of_sources,
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
