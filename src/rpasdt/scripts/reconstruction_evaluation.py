# from rpasdt.algorithm.models import (
#     DiffusionModelSimulationConfig,
#     DiffusionSimulationConfig,
# )
# from rpasdt.algorithm.source_selection import select_sources_with_params
# from rpasdt.algorithm.taxonomies import (
#     DiffusionTypeEnum,
#     SourceSelectionOptionEnum,
# )
# from rpasdt.scripts.taxonomies import graphs
#
# coverages = [40, 60, 80]
# removal_ratio = [5, 10, 15, 20]
#
#
# def evaluation():
#     for graph_function in graphs:
#         graph = graph_function()
#         for coverage in coverages:
#             sources = select_sources_with_params(
#                 graph=graph,
#                 number_of_sources=10,
#                 algorithm=SourceSelectionOptionEnum.BETWEENNESS,
#             )
#             diffusion_config = DiffusionSimulationConfig(
#                 graph=graph,
#                 source_nodes=sources,
#                 iteration_bunch=50,
#                 number_of_experiments=1,
#                 diffusion_models=[
#                     DiffusionModelSimulationConfig(
#                         diffusion_model_type=DiffusionTypeEnum.SI,
#                         diffusion_model_params={"beta": 0.05},
#                     )
#                 ],
#             )
#          pass
