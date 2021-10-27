# Sample code for simulation and evaluation rumor diffusion in the network

Presented simulation is performed for:

- two diffusion model - `SI` with beta = 0.88 and `SIS` with beta = 0.88 and lambda = 0.005
- the experiment will be run under Watts-Strogatz topology with n=100, k = 4 and p =0.5
- there will be 5 randomly selected sources
- for each diffusion model tehre will be 10 experiments
- as a result the average number of required iterations to cover all network is presented

```python
from rpasdt.algorithm.models import DiffusionModelSimulationConfig,
    DiffusionSimulationConfig, NetworkSourceSelectionConfig
from rpasdt.algorithm.simulation import perform_diffusion_simulation
from rpasdt.algorithm.taxonomies import DiffusionTypeEnum, NodeStatusEnum,
    GraphTypeEnum, SourceSelectionOptionEnum

result = perform_diffusion_simulation(DiffusionSimulationConfig(
    number_of_experiments=10,
    graph_type=GraphTypeEnum.WATTS_STROGATZ,
    graph_type_properties={"n": 100, "k": 4, "p": 0.5},
    source_selection_config=NetworkSourceSelectionConfig(number_of_sources=5,
                                                         algorithm=SourceSelectionOptionEnum.RANDOM),
    diffusion_models=[
        DiffusionModelSimulationConfig(
            diffusion_model_type=DiffusionTypeEnum.SI,
            diffusion_model_params={"beta": 0.088}
        ),
        DiffusionModelSimulationConfig(
            diffusion_model_type=DiffusionTypeEnum.SIS,
            diffusion_model_params={"beta": 0.088, "lambda": 0.005}
        )
    ]
))
print(result.avg_iteration_to_status_in_population(NodeStatusEnum.INFECTED))
```

The sample result of the above code:

```
{'SI': 31.4, 'SIS': 40.4}
```

# Sample code for simulation and evaluation for source detection problem

Presented simulation is performed for:

- one diffusion model - `SI` with beta ~ 0.88
- random source selection model with the constant number of selected sources - 5
- sources will be evaluated after 20 simulation iterations
- the source detection will be evaluated with two algorithms: `Netlseuth` and `Rumor center`
- then evaluation of the mentioned algorithms will be presented

```python
from rpasdt.algorithm.models import DiffusionModelSimulationConfig,

SourceDetectionSimulationConfig, NetworkSourceSelectionConfig,
SourceDetectorSimulationConfig, CommunitiesBasedSourceDetectionConfig
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import DiffusionTypeEnum,

SourceDetectionAlgorithm

result = perform_source_detection_simulation(SourceDetectionSimulationConfig(

    diffusion_models=[DiffusionModelSimulationConfig(
        diffusion_model_type=DiffusionTypeEnum.SI,
        diffusion_model_params={"beta": 0.08784399402913001}
    )],
    iteration_bunch=20,
    source_selection_config=NetworkSourceSelectionConfig(
        number_of_sources=5,
    ),
    source_detectors={
        "NETLSEUTH": SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.NET_SLEUTH,
            config=CommunitiesBasedSourceDetectionConfig()
        ),
        "RUMOR_CENTER": SourceDetectorSimulationConfig(
            alg=SourceDetectionAlgorithm.RUMOR_CENTER,
            config=CommunitiesBasedSourceDetectionConfig()
        )
    }
))
print(result.aggregated_results)
for name, results in result.raw_results.items():
    for mm_r in results:
        print(
            f'{name}-{mm_r.real_sources}-{mm_r.detected_sources}-{mm_r.TP}:{mm_r.FN}:{mm_r.FP}')
```

The sample result of the above code:

```
{
    'NETLSEUTH': ExperimentSourceDetectionEvaluation(
        avg_error_distance=6.4,
        recall=0.2,
        precision=0.2,
        f1score=0.20000000000000004
    ),
    'RUMOR_CENTER': ExperimentSourceDetectionEvaluation(
        avg_error_distance=4.4,
        recall=0.16,
        precision=0.16,
        f1score=0.16
     )
}
NETLSEUTH-[28, 14, 15, 10, 9]-[23, 3, 24, 10, 9]-2:3:3
NETLSEUTH-[7, 19, 12, 20, 11]-[22, 3, 10, 31, 11]-1:4:4
NETLSEUTH-[15, 11, 26, 8, 0]-[18, 3, 10, 31, 9]-0:5:5
NETLSEUTH-[32, 0, 16, 33, 8]-[23, 3, 24, 10, 9]-0:5:5
NETLSEUTH-[3, 7, 16, 7, 25]-[22, 3, 24, 10, 9]-1:4:4
NETLSEUTH-[10, 29, 25, 12, 18]-[23, 3, 10, 31, 11]-1:4:4
NETLSEUTH-[8, 24, 22, 19, 10]-[23, 3, 24, 10, 9]-2:3:3
NETLSEUTH-[33, 20, 28, 10, 13]-[23, 3, 24, 10, 9]-1:4:4
NETLSEUTH-[9, 14, 25, 22, 7]-[23, 12, 24, 4, 9]-1:4:4
NETLSEUTH-[19, 13, 12, 27, 17]-[23, 13, 24, 9, 11]-1:4:4
RUMOR_CENTER-[28, 14, 15, 10, 9]-[33, 0, 24, 6, 9]-1:4:4
RUMOR_CENTER-[7, 19, 12, 20, 11]-[33, 0, 6, 24, 11]-1:4:4
RUMOR_CENTER-[15, 11, 26, 8, 0]-[33, 0, 5, 28, 9]-1:4:4
RUMOR_CENTER-[32, 0, 16, 33, 8]-[33, 0, 31, 5, 9]-2:3:3
RUMOR_CENTER-[3, 7, 16, 7, 25]-[33, 0, 24, 6, 9]-0:5:5
RUMOR_CENTER-[10, 29, 25, 12, 18]-[33, 0, 6, 31, 11]-0:5:5
RUMOR_CENTER-[8, 24, 22, 19, 10]-[33, 0, 24, 6, 9]-1:4:4
RUMOR_CENTER-[33, 20, 28, 10, 13]-[33, 0, 31, 5, 9]-1:4:4
RUMOR_CENTER-[9, 14, 25, 22, 7]-[33, 0, 24, 10, 9]-1:4:4
RUMOR_CENTER-[19, 13, 12, 27, 17]-[33, 0, 31, 9, 11]-0:5:5
```
