import csv
import math

from rpasdt.algorithm.models import (
    CentralityBasedSourceDetectionConfig,
    CentralityCommunityBasedSourceDetectionConfig,
    CommunitiesBasedSourceDetectionConfig,
    DiffusionModelSimulationConfig,
    NetworkSourceSelectionConfig,
    SourceDetectionSimulationConfig,
    SourceDetectorSimulationConfig,
    UnbiasedCentralityBasedSourceDetectionConfig,
    UnbiasedCentralityCommunityBasedSourceDetectionConfig,
)
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import (
    CentralityOptionEnum,
    DiffusionTypeEnum,
    SourceDetectionAlgorithm,
    SourceSelectionOptionEnum,
)
from rpasdt.scripts.taxonomies import communities, graphs, sources_number

sir_config = DiffusionModelSimulationConfig(
    diffusion_model_type=DiffusionTypeEnum.SIR,
    diffusion_model_params={"beta": 0.01, "gamma": 0.005},
)

sd_centralities = [
    # CentralityOptionEnum.DEGREE,
    CentralityOptionEnum.BETWEENNESS,
    # CentralityOptionEnum.CLOSENESS,
    # CentralityOptionEnum.EDGE_BETWEENNESS,
    # CentralityOptionEnum.EIGENVECTOR
]

source_detectors = {}

source_detectors.update(
    {
        f"centrality:{centrality}": lambda x, centrality=centrality: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.CENTRALITY_BASED,
            config=CentralityBasedSourceDetectionConfig(
                number_of_sources=x, centrality_algorithm=centrality
            ),
        )
        for centrality in sd_centralities
    }
)

source_detectors.update(
    {
        f"unbiased:{centrality}": lambda x, centrality=centrality: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.UNBIASED_CENTRALITY_BASED,
            config=UnbiasedCentralityBasedSourceDetectionConfig(
                number_of_sources=x, centrality_algorithm=centrality
            ),
        )
        for centrality in sd_centralities
    }
)

source_detectors.update(
    {
        f"centrality-cm:{centrality}:{cm}": lambda x, centrality=centrality, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_CENTRALITY_BASED,
            config=CentralityCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                centrality_algorithm=centrality,
                communities_algorithm=cm,
            ),
        )
        for centrality in sd_centralities
        for cm in communities
    }
)

source_detectors.update(
    {
        f"unbiased-cm:{centrality}:{cm}": lambda x, centrality=centrality, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.COMMUNITY_UNBIASED_CENTRALITY_BASED,
            config=UnbiasedCentralityCommunityBasedSourceDetectionConfig(
                number_of_sources=x,
                centrality_algorithm=centrality,
                communities_algorithm=cm,
            ),
        )
        for centrality in sd_centralities
        for cm in communities
    }
)
source_detectors.update(
    {
        f"rumor:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.RUMOR_CENTER,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
)
source_detectors.update(
    {
        f"jordan:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.JORDAN_CENTER,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
)

source_detectors.update(
    {
        f"netsleuth:{cm}": lambda x, cm=cm: SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.NET_SLEUTH,
            config=CommunitiesBasedSourceDetectionConfig(
                number_of_sources=x, communities_algorithm=cm
            ),
        )
        for cm in communities
    }
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
    filename: str = f"results/sd/{graph_function.__name__}.csv"
    file = open(filename, "w")
    csvwriter = csv.writer(file)
    csvwriter.writerow(header)
    file.close()

    graph = graph_function()
    for sn in sources_number:
        source_number = sn if sn > 1 else max(2, math.ceil(len(graph.nodes) * sn))
        sd_local = {
            name: config(source_number) for name, config in source_detectors.items()
        }
        source_detection_config = SourceDetectionSimulationConfig(
            number_of_experiments=3,
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
