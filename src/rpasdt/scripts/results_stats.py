import csv
import os

from rpasdt.algorithm.taxonomies import CommunityOptionEnum

METHODS = [
    CommunityOptionEnum.LOUVAIN,
    CommunityOptionEnum.BELIEF,
    CommunityOptionEnum.LEIDEN,
    CommunityOptionEnum.LABEL_PROPAGATION,
    CommunityOptionEnum.GREEDY_MODULARITY,
    CommunityOptionEnum.EIGENVECTOR,
    CommunityOptionEnum.GA,
    CommunityOptionEnum.INFOMAP,
    CommunityOptionEnum.KCUT,
    CommunityOptionEnum.MARKOV_CLUSTERING,
    CommunityOptionEnum.PARIS,
    CommunityOptionEnum.SPINGLASS,
    CommunityOptionEnum.SURPRISE_COMMUNITIES,
    CommunityOptionEnum.WALKTRAP,
    CommunityOptionEnum.SPECTRAL,
    CommunityOptionEnum.SBM_DL,
]


def draw_passed_computations():
    for filename in os.listdir("results"):
        if "ce" in filename:
            with open(f"results/{filename}", newline="\n") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",")
                for row in spamreader:
                    print(", ".join(row))


draw_passed_computations()
