import csv
import math

from rpasdt.algorithm.models import (
    DiffusionModelSimulationConfig,
    NetworkSourceSelectionConfig,
    SourceDetectionSimulationConfig,
)
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import (
    DiffusionTypeEnum,
    SourceSelectionOptionEnum,
)
from rpasdt.scripts.taxonomies import graphs, source_detectors, sources_number

sir_config = DiffusionModelSimulationConfig(
    diffusion_model_type=DiffusionTypeEnum.SIR,
    diffusion_model_params={"beta": 0.01, "gamma": 0.005},
)

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
for graph_function in graphs:
    filename: str = f"results/sd_df/{graph_function.__name__}.csv"
    file = open(filename, "w")
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    file.close()
    print(f"Processing {graph_function.__name__}")
    graph = graph_function()
    for sn in sources_number:
        source_number = sn if sn > 1 else max(2, math.ceil(len(graph.nodes) * sn))
        sd_local = {
            name: config(source_number) for name, config in source_detectors.items()
        }
        source_detection_config = SourceDetectionSimulationConfig(
            number_of_experiments=1,
            diffusion_models=[sir_config],
            iteration_bunch=50,
            source_selection_config=NetworkSourceSelectionConfig(
                algorithm=SourceSelectionOptionEnum.BETWEENNESS,
                number_of_sources=source_number,
            ),
            graph=graph,
            source_detectors=sd_local,
            timeout=90,
        )
        result = perform_source_detection_simulation(
            source_detection_config=source_detection_config
        )

        for config, rr in result.aggregated_results.items():
            row = [
                config,
                source_number,
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
