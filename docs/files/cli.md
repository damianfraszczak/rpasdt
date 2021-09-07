# CLI Commands
This file contains the sample usage of the provided CLI commands by RP&SDT.
To run any of the command please use the following syntax
```
docker-compose run rpasdt cli <command>
```
to get help for the given command use
```
docker-compose run rpasdt cli <command> help
```

List of the available commands

     compute_centrality
       Compute centrality for the given graph.

     compute_communities
       Compute communities for the given graph.

     diffusion_simulation_experiment
       Perform diffusion simulation experiment.

     generate_graph
       Generate graph.

     simulate_diffusion
       Simulate diffusion with selected model under given network.

     source_detection_experiment
       Perform diffusion simulation experiment.

**NOTE**
While passing dicts as kwargs to the commands please don't pass them with spaces so instead of `{"a": 1}` you should pass `{"a":1}` as otherwise they will parsed incorrectly.

## Print general help
```
docker-compose run rpasdt cli help
```

## Generate network
```
docker-compose run rpasdt cli generate_graph --graph_type=KARATE_CLUB --output_file_path==output.txt
```

## Compute centrality for the given network.

```
docker-compose run rpasdt cli compute_centrality --input_graph_path=output.txt --centrality=DEGREE
```

## Find communities in the given network.

```
docker-compose run rpasdt cli compute_communities --input_graph_path=output.txt --community=LOUVAIN
```

## Perform a diffusion simulation
```
docker-compose run rpasdt cli simulate_diffusion karate.json SI --source_nodes=1,2,3 --output_file_path=karate_diffusion.json --model_params=
{'beta':0.8}
``````

## Perform a diffusion simulation experiment
```
docker-compose run rpasdt cli diffusion_simulation_experiment --config_file_path=experiment.json --output_file_path=experiment-output.json
``````

## Perform a source detection experiment
```
docker-compose run rpasdt cli source_detection_experiment --config_file_path=experiment.json --output_file_path=experiment-output.json
``````
