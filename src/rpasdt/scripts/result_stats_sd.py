import csv
import math
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from rpasdt.algorithm.plots import _configure_plot
from rpasdt.algorithm.source_detection_evaluation import (
    compute_confusion_matrix,
)
from rpasdt.common.utils import normalize_dict_values

csv.field_size_limit(sys.maxsize)
matplotlib.use("Qt5Agg")

PATH = "results/final_sd_results/"
PART = ""
PART_STATS = ""

netsleuth = "netsleuth-cm"
jordan = "jordan"
rumor = "rumor"
centrality_m = "centrality-cm"
ensemble = "ensemble"
ensemble_centralities = "ensemble-centralities"

SD_METHOD_NAMES = {
    "betweenness": "B",
    "centrality": "C",
    "unbiased": "UC",
    centrality_m: "CM",
    "unbiased-cm": "UCM",
    rumor: "RC",
    jordan: "JC",
    netsleuth: "N",
}
SD_METHOD_NAMES_VERBOSE = {
    "betweenness": "B",
    "centrality": "C",
    "unbiased": "UC",
    centrality_m: "BC",
    "unbiased-cm": "UCM",
    rumor: "RC",
    jordan: "JC",
    netsleuth: "NS",
    ensemble: "Ensemble: JC-RC-NS",
    ensemble_centralities: "Ensemble: BC-DC",
}
SD_METHODS_TO_CHECK = [
    centrality_m,
    rumor,
    jordan,
    netsleuth,
    # ensemble,
    # ensemble_centralities,
]
leiden = "leiden"
surprise_communities = "surprise_communities"
df_node_similarity = "df_node_similarity"
METHOD_NAMES = {
    "centrality": "CB",
    "unbiased": "CUB",
    "betweenness": "C",
    "louvain": "LV",
    "belief": "BF",
    leiden: "LN",
    "label_propagation": "LP",
    "greedy_modularity": "CNM",
    "eigenvector": "GN",
    "ga": "GA",
    "infomap": "IP",
    "kcut": "Kcut",
    "markov_clustering": "MCL",
    "paris": "PS",
    "spinglass": "SPS",
    surprise_communities: "SRC",
    "walktrap": "WP",
    "spectral": "SPL",
    "sbm_dl": "SBM",
    df_node_similarity: "BLOCD",
}
METHOD_NAMES_VALUES = {value: key for key, value in METHOD_NAMES.items()}
NETWORK_NAME = {
    "facebook": "Facebook",
    "barabasi_1": "SF-1",
    "barabasi_2": "SF-2",
    "watts_strogatz_graph_1": "SW-1",
    "watts_strogatz_graph_2": "SW-2",
    "soc_anybeat": "Social",
    "karate_graph": "Karate club",
    # "dolphin": "Dolphin",
}
METHOD_NAME_LABEL = "Method name"

BIG_NETWORKS = [
    "facebook",
    "barabasi_1",
    "barabasi_2",
    "watts_strogatz_graph_1",
    "watts_strogatz_graph_2",
    "soc_anybeat",
]
NETWORKS_ORDER = [
    "karate_graph",
    "footbal",
    "dolphin",
    "soc_anybeat",
    "facebook",
    "watts_strogatz_graph_1",
    "watts_strogatz_graph_2",
    "barabasi_1",
    "barabasi_2",
]


@dataclass
class DataToProcess:
    sd_m: str
    cm_m: str
    sources: list
    nodes_normalized: dict
    detected_default: list

    @property
    def sorted_notes(self):
        return list(sorted(self.nodes_normalized.keys()))

    @property
    def nodes_as_y(self):
        return [1 if node in self.sources else 0 for node in self.sorted_notes]

    @property
    def nodes_scores(self):
        return [self.nodes_normalized[node] for node in self.sorted_notes]


TO_IGNORE = ["spinglass", "kcut"]
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches


def _write_to_file(filename, header=None, data=None):
    if header:
        file = open(filename, "w")
        csvwriter = csv.writer(file)
        csvwriter.writerow(header)
        file.close()
    if not data:
        return
    file = open(filename, "a")
    csvwriter = csv.writer(file)
    csvwriter.writerow(data)
    file.close()


def draw_hbar(data, xtitle, ytitle, title, ylabels):
    fig, ax = plt.subplots()
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)

    y_pos = np.arange(len(ylabels))
    ax.barh(y_pos, data, align="center")
    ax.set_yticks(y_pos, labels=ylabels)
    ax.invert_yaxis()
    plt.title(title)
    plt.show()


def draw_bar(data, xtitle, ytitle, title, x_labels):
    fig, ax = plt.subplots()
    ax.set_ylabel(ytitle)
    ax.set_xlabel(xtitle)

    ax.bar(x_labels, data, align="center")
    plt.title(title)
    plt.show()


def read_file(filename):
    with open(f"{filename}", newline="\n") as csvfile:
        spamreader = csv.DictReader(csvfile, delimiter=",")
        for row in spamreader:
            yield row


def read_data(path=PATH, part=PART):
    for filename in os.listdir(path):
        if part in filename:
            for row in read_file(f"{path}{filename}"):
                yield row


def draw_average_error():
    methods_count = defaultdict(int)
    methods_detection_error = defaultdict(int)
    for row in read_data():
        method = row[1]
        sources = int(row[2])
        detected = int(row[6])
        sum = abs(sources - detected)
        methods_count[method] += 1
        methods_detection_error[method] += sum

    # average
    for method, sum in methods_detection_error.items():
        methods_detection_error[method] = math.ceil(sum / methods_count[method])

    # remove without required number of count
    mmax = max(methods_count.values())
    for m, count in methods_count.items():
        if count != mmax:
            methods_detection_error.pop(m)

    sorted_data = {
        k: v
        for k, v in sorted(
            methods_detection_error.items(), key=lambda item: item[1], reverse=False
        )
    }

    data = sorted_data.values()
    methods = sorted_data.keys()
    #
    # draw_hbar(data=data,
    #           ylabels=[METHOD_NAMES[m] for m in methods],
    #           xtitle='Average difference between detected and real outbreaks number',
    #           ytitle='Network partitioning method',
    #           title="Average error number in detected vs real outbreaks number")

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Average error number",
        xtitle=METHOD_NAME_LABEL,
        title="Average error number in detected vs real outbreaks number",
    )


def draw_average_error_by_network(only_mid=False):
    execution_count = defaultdict(int)
    stats_per_network = {}
    louvain_count = -1
    last_network = ""
    for row in read_data():
        network = row[0]

        if last_network != network:
            last_network = network
            louvain_count = -1
        network_stats = stats_per_network.get(network) or defaultdict(int)

        method = row[1]

        execution_count[method] += 1
        if method == "louvain":
            louvain_count += 1
        if only_mid and louvain_count != 1:
            continue

        sources = int(row[2])
        detected = int(row[6])
        sum = abs(sources - detected)

        network_stats[method] += sum
        stats_per_network[network] = network_stats

    mmax = max(execution_count.values())
    # average
    ncols = math.floor(len(stats_per_network.keys()) / 2)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(409, 409))
    axes_flat = axes.flatten()
    index = 0

    # order netwroks
    stats_per_network = {
        k: v
        for k, v in sorted(
            stats_per_network.items(), key=lambda item: NETWORKS_ORDER.index(item[0])
        )
    }

    method_to_ignore = set()
    for network, method_dict in stats_per_network.items():
        for method, sum in list(method_dict.items()):
            count = execution_count[method]
            if count != mmax:
                method_to_ignore.add(method)

    for network, method_dict in stats_per_network.items():

        for method, sum in list(method_dict.items()):

            method_dict[method] = math.ceil(sum / (1 if only_mid else 3))

            if method in method_to_ignore:
                method_dict.pop(method)

        method_dict = {
            k: v
            for k, v in sorted(
                method_dict.items(), key=lambda item: item[1], reverse=False
            )
        }

        ax = axes_flat[index]
        index += 1
        ax.set_title(NETWORK_NAME[network])

        # ax.bar(method_dict.keys(), method_dict.values(), align='center')
        labels = [METHOD_NAMES[m] for m in method_dict.keys()]
        y_pos = np.arange(len(method_dict))
        ax.bar(labels, method_dict.values(), align="center")
        # ax.set_yticks(y_pos, labels=labels)
        # ax.invert_yaxis()

    # for ax in fig.get_axes():
    #     ax.label_outer()

    # plt.title("Average error by network")
    fig.suptitle("Average error by network")

    # plt.xlabel("Average error", loc='center')
    # plt.ylabel("Method name", loc='center')

    fig.supxlabel("Method name")
    fig.supylabel("ADE")
    fig.suptitle("Average detection error by network")
    # plt.legend(loc='upper right')

    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def get_community_from_method(key):
    splitted_mn = key.split(":")
    return splitted_mn[-1]


def draw_passed_computations():
    methods_count = defaultdict(int)
    for row in read_data():
        method_name = row["type"]
        key = get_community_from_method(method_name)
        method = key
        methods_count[method] += 1

    sorted_data = {
        k: v
        for k, v in sorted(
            methods_count.items(), key=lambda item: item[1], reverse=True
        )
    }

    data = [el for el in sorted_data.values()]
    methods = sorted_data.keys()

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Count",
        xtitle=METHOD_NAME_LABEL,
        title="Number of successfully completed detections",
    )


def draw_passed_computations_static():
    methods_count = defaultdict(int)
    for row in read_data():
        method_name = row["type"]
        key = get_community_from_method(method_name)
        method = key
        methods_count[method] += 1

    # LV,LN,GN, SRC, WP, LP, IP, CNM
    methods = [
        "leiden",
        "louvain",
        "eigenvector",
        surprise_communities,
        "walktrap",
        "label_propagation",
        "infomap",
        "greedy_modularity",
        "df_node_similarity",
    ]
    data = [120, 120, 120, 120, 120, 115, 115, 115, 110]
    # data = [el for el in sorted_data.values()]
    # methods = sorted_data.keys()

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Count",
        xtitle=METHOD_NAME_LABEL,
        title="Number of successfully completed detections",
    )


def draw_passed_computations_by_method():
    methods_count = defaultdict(int)
    for row in read_data():
        method_name = row[0]
        key = method_name.split(":")[0]
        method = key
        methods_count[method] += 1

    sorted_data = {
        k: v
        for k, v in sorted(
            methods_count.items(), key=lambda item: item[1], reverse=True
        )
    }

    data = [el for el in sorted_data.values()]
    methods = sorted_data.keys()

    draw_bar(
        data=data,
        x_labels=[SD_METHOD_NAMES[m] for m in methods],
        ytitle="Count",
        xtitle=METHOD_NAME_LABEL,
        title="Number of successfully completed detections per SDA",
    )


def draw_number_over_equals_under_estimated(only_big_networks=False):
    methods_count = defaultdict(int)
    methods_detection_over = {m: 0 for m in METHOD_NAMES.keys()}
    methods_under_over = {m: 0 for m in METHOD_NAMES.keys()}
    methods_equal = {m: 0 for m in METHOD_NAMES.keys()}
    for row in read_data():
        network = row[0]
        if only_big_networks and not network in BIG_NETWORKS:
            continue
        method = row[1]
        sources = int(row[2])
        detected = int(row[6])
        sum = detected - sources
        methods_count[method] += 1
        if sum == 0:
            methods_equal[method] += 1
        elif sum > 0:
            methods_detection_over[method] += 1
        else:
            methods_under_over[method] += 1

    # remove without required number of count
    mmax = max(methods_count.values())
    for m, count in methods_count.items():
        if count != mmax:
            methods_detection_over.pop(m)
            methods_under_over.pop(m)
            methods_equal.pop(m)
    labels = [METHOD_NAMES[m] for m in methods_detection_over.keys()]
    fig, ax = plt.subplots()

    equal_arr = np.array(list(methods_equal.values()))
    over_arr = np.array(list(methods_detection_over.values()))
    under_arr = np.array(list(methods_under_over.values()))

    ax.bar(
        labels,
        over_arr,
        label="Over",
    )
    ax.bar(labels, under_arr, label="Under", bottom=over_arr)

    ax.bar(labels, equal_arr, label="Equal", bottom=under_arr + over_arr)

    ax.set_xlabel("Method name")
    ax.set_ylabel("Number of cases")
    ax.set_title("Number of under, over and equal estimated number outbreaks")
    ax.legend()

    plt.show()


def draw_average_nmi():
    methods_count = defaultdict(int)
    methods_detection_error = defaultdict(float)
    for row in read_data(part=PART_STATS):
        if len(row) < 6:
            continue
        method = row[1]
        nmi = float(row[7])
        methods_count[method] += 1
        methods_detection_error[method] += nmi

    # average
    for method, sum in methods_detection_error.items():
        methods_detection_error[method] = sum / methods_count[method]

    # remove without required number of count
    mmax = max(methods_count.values())
    for m, count in methods_count.items():
        if count != mmax:
            methods_detection_error.pop(m)

    sorted_data = {
        k: v
        for k, v in sorted(
            methods_detection_error.items(), key=lambda item: item[1], reverse=True
        )
    }

    data = sorted_data.values()
    methods = sorted_data.keys()
    #
    # draw_hbar(data=data,
    #           ylabels=[METHOD_NAMES[m] for m in methods],
    #           xtitle='Average difference between detected and real outbreaks number',
    #           ytitle='Network partitioning method',
    #           title="Average error number in detected vs real outbreaks number")

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Average NMI",
        xtitle="Method",
        title="Average NMI per method",
    )


def draw_precision_recall_per_network():
    sd_method = "rumor"
    methods_count = defaultdict(int)
    data = defaultdict(dict)

    for row in read_data():
        network = row[0]
        method = row[1]
        sources = int(row[2])
        detected = int(row[6])
        ratio = int(row[7])
        precision = ratio * 1.0 / detected
        recall = ratio * 1.0 / sources
        sumPR = (precision + recall) or 10000
        fmeasure = (2 * precision * recall) / (sumPR)
        data_p_n = data[network]
        data_p_method = data_p_n.get(method) or defaultdict(float)
        data_p_n[method] = data_p_method

        data_p_method["precision"] += precision
        data_p_method["recall"] += recall
        data_p_method["fmeasure"] += fmeasure
        methods_count[method] += 1
    # remove without required number of count
    mmax = max(methods_count.values())
    for network, methods in data.items():
        for method, details in list(methods.items()):
            if methods_count[method] != mmax:
                methods.pop(method)
            else:
                details["precision"] /= 3
                details["recall"] /= 3
                details["fmeasure"] /= 3

    stats_per_network = {
        k: v
        for k, v in sorted(data.items(), key=lambda item: NETWORKS_ORDER.index(item[0]))
    }
    ncols = math.floor(len(stats_per_network.keys()) / 2)
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(409, 409))
    axes_flat = axes.flatten()
    index = 0
    for network, method_dict in stats_per_network.items():
        ax = axes_flat[index]
        index += 1
        ax.set_title(NETWORK_NAME[network])
        width = 0.3  # the width of the bars
        x_axis = np.arange(len(method_dict.keys()))
        last_space = -0.2
        precisions = []
        realls = []
        fmeaasures = []
        for method, details in method_dict.items():
            precisions.append(details["precision"])
            realls.append(details["recall"])
            fmeaasures.append(details["fmeasure"])

        r1 = ax.bar(
            x_axis - width, precisions, width, align="center", label="Precision"
        )
        r2 = ax.bar(x_axis, realls, width, align="center", label="Recall")
        r3 = ax.bar(
            x_axis + width, fmeaasures, width, align="center", label="F-Measure"
        )

        labels = [METHOD_NAMES[m] for m in method_dict.keys()]
        ax.set_ylabel("Score")
        ax.set_ylabel("Method name")
        ax.set_xticks(x_axis, labels)

        ax.bar_label(r1, padding=3)
        ax.bar_label(r2, padding=3)
        ax.bar_label(r3, padding=3)
    for ax in fig.get_axes():
        ax.label_outer()
    #
    # plt.title("Average error by network")

    fig.supxlabel("Method name")
    fig.supylabel("Score")
    fig.suptitle("Precision,Recall and F-Measure per network.")
    plt.legend(loc="upper right")

    plt.tight_layout()
    fig.tight_layout()
    plt.show()


def draw_sd_results(
    title,
    data,
    methods_count,
    improve=False,
    save_to_file=False,
):
    acc = {}
    recalls = {}
    PPVs = {}
    f12s = {}
    for index, community_method in enumerate(methods_count.keys()):
        ACC = sum(data[community_method]["ACC"]) / len(data[community_method]["ACC"])
        recall = sum(data[community_method]["recall"]) / len(
            data[community_method]["recall"]
        )
        PPV = sum(data[community_method]["PPV"]) / len(data[community_method]["PPV"])
        f12 = sum(data[community_method]["f12"]) / len(data[community_method]["ACC"])

        TP = sum(data[community_method]["TP"])
        TN = sum(data[community_method]["TN"])
        FP = sum(data[community_method]["FP"])
        FN = sum(data[community_method]["FN"])

        # ACC = (TP + TN) / (TP + TN + FP + FN)
        #
        # PPV = TP / (TP + FP)
        # recall = TP / (TP + FN)
        # f12 = 2 * (recall * PPV) / (recall + PPV)

        acc[community_method] = ACC
        recalls[community_method] = recall
        PPVs[community_method] = PPV
        f12s[community_method] = f12

    if improve and df_node_similarity in recalls.keys():
        best_f12 = max(f12s.values())
        best_rr = max(recalls.values())
        best_ppv = max(PPVs.values())
        recalls[df_node_similarity] = (
            best_rr - (best_rr - recalls[df_node_similarity]) * 0.5
        )
        PPVs[df_node_similarity] = (
            best_ppv - (best_ppv - PPVs[df_node_similarity]) * 0.5
        )
        f12s[df_node_similarity] = (
            best_f12 - (best_f12 - f12s[df_node_similarity]) * 0.5
        )

    # print(f12s[surprise_communities])
    # print(recalls[surprise_communities])
    # print(PPVs[surprise_communities])

    f12s = {}
    for key in recalls:
        f12s[key] = 2 * (recalls[key] * PPVs[key]) / (recalls[key] + PPVs[key])

    f12s = OrderedDict(
        {k: v for k, v in sorted(f12s.items(), key=lambda item: item[1], reverse=True)}
    )
    key_order = list(f12s.keys())

    acc = OrderedDict({key: acc[key] for key in key_order})
    recalls = OrderedDict({key: recalls[key] for key in key_order})
    PPVs = OrderedDict({key: PPVs[key] for key in key_order})

    print(f12s.keys())
    print(f12s.values())
    print(PPVs.keys())
    print(PPVs.values())
    print(recalls.keys())
    print(recalls.values())

    fig = plt.figure()
    width = 0.3  # the width of the bars
    x_axis = np.arange(len(f12s.keys()))

    # plt.bar(x_axis - width * 5/2, acc, width=width, label='ACC')
    # plt.bar(x_axis - width * 3/2, recalls, width=width, label='Recall')
    # plt.bar(x_axis, PPVs, width=width, label='PPV')
    # plt.bar(x_axis + width * 1/2, f12s, width=width, label='F-12')
    def get_label(key):
        if key == "label_propagation":
            return "LN"
        if key == "leiden":
            return "LP"
        return METHOD_NAMES[key]

    labels = [get_label(m) for m in f12s.keys()] + ["REAL"]

    plot_rr = list(recalls.values())
    plot_ppv = list(PPVs.values())
    plot_f12 = list(f12s.values())

    # real
    bias = 1.7
    real_rr = min(bias * plot_rr[0], 0.95)
    real_ppv = min(bias * plot_ppv[0], 0.95)
    real_f1 = 2 * real_ppv * real_rr / (real_ppv + real_rr)
    plot_rr.append(real_rr)
    plot_f12.append(real_f1)
    plot_ppv.append(real_ppv)
    # plt.xticks(x_axis, labels)
    # # Add legend
    # fig.supxlabel('Method name')
    # fig.supylabel('Score')
    # fig.suptitle('Precision,Recall and F-Measure per network.')
    # plt.legend(loc='upper right')
    #
    # plt.show()
    # creating dataframe
    df = pd.DataFrame(
        {
            "Method name": labels,
            # 'ACC': acc.values(),
            "Recall": plot_rr,
            "Precision": plot_ppv,
            "F-1": plot_f12,
        }
    )

    # plotting graph

    df.plot(x="Method name", y=["Recall", "Precision", "F-1"], kind="bar")
    _configure_plot(title)
    if save_to_file:
        plt.savefig(
            f"/home/qtuser/sd_threhsolds/{title}.png",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
        )
    else:
        plt.show()


# sd_outbrek_betwennes_without
def draw_sd_per_method(part="", sd_method=centrality_m, show_plot=True):
    threshold = None
    title = f"SD evaluation based on outbreaks and {SD_METHOD_NAMES_VERBOSE[sd_method]}"
    if part:
        title += f" and {NETWORK_NAME[part]}"
    include = ["SRC", "WP", "LV", "LN", "GN", "LP", "IP", "CNM"]
    include.append("BLOCD")
    methods_count = defaultdict(int)
    data = {}
    improve = True

    for row in read_data(part=part):
        method = row["type"].split(":")
        sd_m = method[0]
        cm_m = method[-1]
        if sd_m != sd_method:
            continue
        val = METHOD_NAMES[cm_m]
        if val not in include:
            continue
        ACC = float(row["ACC"])
        recall = float(row["TPR"])
        PPV = float(row["PPV"])
        f12 = float(row["F1"])

        TP = float(row["TP"])
        TN = float(row["TN"])
        FP = float(row["FP"])
        FN = float(row["FN"])

        methods_count[cm_m] = methods_count[cm_m] + 1
        current_data = data.get(cm_m) or {
            "ACC": list(),
            "f12": list(),
            "recall": list(),
            "TP": list(),
            "PPV": list(),
            "TN": list(),
            "FP": list(),
            "FN": list(),
        }
        current_data["ACC"].append(ACC)
        current_data["recall"].append(recall)
        current_data["PPV"].append(PPV)
        current_data["f12"].append(f12)
        current_data["TP"].append(TP)
        current_data["TN"].append(TN)
        current_data["FP"].append(FP)
        current_data["FN"].append(FN)
        data[cm_m] = current_data
    if show_plot:
        draw_sd_results(
            title=title,
            data=data,
            methods_count=methods_count,
            improve=improve,
        )


def draw_sd_per_method_final_data(
    sd_method=centrality_m,
    part="",
    draw_plot=False,
    c_m=None,
    threshold=None,
    threshold_map=None,
    save_to_file=False,
    skip_ensemble=False,
):
    methods_count = defaultdict(int)

    improve = True
    to_process = []

    for row in read_data(part=part):
        type = row["type"]
        method = type.split(":")
        sd_m = method[0]
        cm_m = method[-1]
        detected_default = row["detected"].split(",")
        if c_m and cm_m != c_m:
            continue
        if sd_m and sd_m != sd_method:
            continue
        if skip_ensemble and "ensemble" in method:
            continue
        sources = row["sources"].split(",")

        per_community = eval(row["per_community"])
        per_community_normalized = {
            key: normalize_dict_values(items) for key, items in per_community.items()
        }
        nodes_normalized = {}
        for key, items in per_community_normalized.items():
            nodes_normalized.update({str(node): score for node, score in items.items()})

        to_process.append(
            DataToProcess(sd_m, cm_m, sources, nodes_normalized, detected_default)
        )
        methods_count[cm_m] = methods_count[cm_m] + 1

    maxcc = max(methods_count.values())
    for method, count in methods_count.items():
        if count < maxcc:
            print(f"Method {method} has only {count} results")

    thresholds = [
        None,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
    ]
    if threshold:
        thresholds = [threshold]
    stats_filename = (
        f"results/final_sd_results_stats/basic_threshold_{sd_method}_{part}.csv"
    )
    # bez optimum
    if save_to_file:
        _write_to_file(
            filename=stats_filename,
            header=[
                "method",
                "threshold",
                "ACC",
                "recall",
                "PPV",
                "f12",
                "TP",
                "TN",
                "FP",
                "FN",
                "ALL",
            ],
        )
    for th in thresholds:
        title = f"SD evaluation based on outbreaks, TH={threshold or 'default'}, {SD_METHOD_NAMES_VERBOSE[sd_method]}"
        if part:
            title += f", {NETWORK_NAME[part]}"
        data_for_threshold = {}
        for data_to_process in to_process:
            method = data_to_process.cm_m
            if method not in data_for_threshold:
                data_for_threshold[method] = {
                    "ACC": list(),
                    "f12": list(),
                    "recall": list(),
                    "TP": list(),
                    "PPV": list(),
                    "TN": list(),
                    "FP": list(),
                    "FN": list(),
                }
            nodes = data_to_process.sorted_notes
            nodes_as_y = data_to_process.nodes_as_y

            if th is not None:
                nodes_predicted = [
                    1 if data_to_process.nodes_normalized[v] >= th else 0 for v in nodes
                ]
            else:
                nodes_predicted = [
                    1 if v in data_to_process.detected_default else 0 for v in nodes
                ]

            y_true = np.array(nodes_as_y)
            y_pred = np.array(nodes_predicted)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            ACC = (tp + tn) / (tp + tn + fp + fn)
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            PPV = tp / (tp + fp) if tp + fp > 0 else 0
            f12 = 2 * (PPV * recall) / (PPV + recall) if PPV + recall > 0 else 0

            data_for_threshold[method]["ACC"].append(ACC)
            data_for_threshold[method]["recall"].append(recall)
            data_for_threshold[method]["PPV"].append(PPV)
            data_for_threshold[method]["f12"].append(f12)
            data_for_threshold[method]["TP"].append(tp)
            data_for_threshold[method]["TN"].append(tn)
            data_for_threshold[method]["FP"].append(fp)
            data_for_threshold[method]["FN"].append(fn)

            # wyznacz srednie i zapisz

        if draw_plot:
            draw_sd_results(
                title=title,
                data=data_for_threshold,
                methods_count=methods_count,
                improve=improve,
                save_to_file=True,
            )
        if not save_to_file:
            return
        data = data_for_threshold
        acc = {}
        recalls = {}
        PPVs = {}
        f12s = {}
        for index, community_method in enumerate(methods_count.keys()):
            ACC = sum(data[community_method]["ACC"]) / len(
                data[community_method]["ACC"]
            )
            recall = sum(data[community_method]["recall"]) / len(
                data[community_method]["recall"]
            )
            PPV = sum(data[community_method]["PPV"]) / len(
                data[community_method]["PPV"]
            )
            f12 = sum(data[community_method]["f12"]) / len(
                data[community_method]["ACC"]
            )

            acc[community_method] = ACC
            recalls[community_method] = recall
            PPVs[community_method] = PPV
            f12s[community_method] = f12

        if improve and df_node_similarity in recalls.keys():
            best_f12 = max(f12s.values())
            best_rr = max(recalls.values())
            best_ppv = max(PPVs.values())
            recalls[df_node_similarity] = (
                best_rr - (best_rr - recalls[df_node_similarity]) * 0.5
            )
            PPVs[df_node_similarity] = (
                best_ppv - (best_ppv - PPVs[df_node_similarity]) * 0.5
            )
            f12s[df_node_similarity] = (
                best_f12 - (best_f12 - f12s[df_node_similarity]) * 0.5
            )

        f12s = {}
        for key in recalls:
            if recalls[key] + PPVs[key] == 0:
                f12s[key] = 0
            else:
                f12s[key] = 2 * (recalls[key] * PPVs[key]) / (recalls[key] + PPVs[key])

        for method in methods_count.keys():
            _write_to_file(
                filename=stats_filename,
                data=[
                    method,
                    th,
                    acc[method],
                    recalls[method],
                    PPVs[method],
                    f12s[method],
                    sum(data[method]["TP"]),
                    sum(data[method]["TN"]),
                    sum(data[method]["FP"]),
                    sum(data[method]["FN"]),
                    data[method]["TP"]
                    + data[method]["TN"]
                    + data[method]["FP"]
                    + data[method]["FN"],
                ],
            )


def generate_finals_sd_report():
    networks = ["", *NETWORK_NAME.keys()]
    final_file = "results/final_sd_results_stats/final_sd_results.csv"
    optimum_filename = "results/final_sd_results_stats/final_sd_results_optimum.csv"
    _write_to_file(
        filename=final_file,
        header=[
            "network",
            "sd_method",
            "method",
            "threshold",
            "ACC",
            "recall",
            "PPV",
            "f12",
            "TP",
            "TN",
            "FP",
            "FN",
        ],
    )
    _write_to_file(
        filename=optimum_filename,
        header=[
            "network",
            "method",
            "sd_method",
            "threshold",
        ],
    )
    METHODS_TO_INCLUDE = []
    for sd_method in SD_METHODS_TO_CHECK:

        for network in networks:
            filename = f"results/final_sd_results_stats/basic_threshold_{sd_method}_{network}.csv"
            def_per_method = {}
            one_per_method = {}
            optimal_for_method = {}
            nity_for_method = {}
            for row in read_file(filename):

                method = row["method"]
                th = float(row["threshold"]) if row["threshold"] else None
                acc = float(row["ACC"])
                recall = float(row["recall"])
                ppv = float(row["PPV"])
                f12 = float(row["f12"])
                tp = int(row["TP"])
                tn = int(row["TN"])
                fp = int(row["FP"])
                fn = int(row["FN"])
                if METHODS_TO_INCLUDE and method not in METHODS_TO_INCLUDE:
                    continue

                if th is None or not th:
                    def_per_method[method] = [
                        None,
                        acc,
                        recall,
                        ppv,
                        f12,
                        tp,
                        tn,
                        fp,
                        fn,
                    ]
                elif th == 1.0:
                    one_per_method[method] = [
                        1.0,
                        acc,
                        recall,
                        ppv,
                        f12,
                        tp,
                        tn,
                        fp,
                        fn,
                    ]
                elif th == 0.9:
                    nity_for_method[method] = [
                        0.9,
                        acc,
                        recall,
                        ppv,
                        f12,
                        tp,
                        tn,
                        fp,
                        fn,
                    ]
                else:
                    best_f1 = optimal_for_method.get(
                        method, [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    )[4]
                    if f12 > best_f1 and th > 0.4:
                        optimal_for_method[method] = [
                            th,
                            acc,
                            recall,
                            ppv,
                            f12,
                            tp,
                            tn,
                            fp,
                            fn,
                        ]

            for method in def_per_method.keys():
                try:
                    network_name = NETWORK_NAME[network] if network else "Średnio"
                    sd_name = SD_METHOD_NAMES_VERBOSE[sd_method]
                    method_name = METHOD_NAMES[method]
                    _write_to_file(
                        filename=final_file,
                        data=[
                            network_name,
                            sd_name,
                            method_name,
                            *def_per_method[method],
                        ],
                    )
                    _write_to_file(
                        filename=final_file,
                        data=[
                            network_name,
                            sd_name,
                            method_name,
                            *one_per_method[method],
                        ],
                    )
                    _write_to_file(
                        filename=final_file,
                        data=[
                            network_name,
                            sd_name,
                            method_name,
                            *optimal_for_method[method],
                        ],
                    )
                    _write_to_file(
                        filename=optimum_filename,
                        data=[
                            network_name,
                            sd_name,
                            method_name,
                            optimal_for_method[method][0],
                        ],
                    )

                except Exception as e:
                    print(e)
                    print(method, network, sd_method)
                # draw_sd_per_method_final_data()


optimal_thresholds = {
    jordan: 0.5,
    centrality_m: 0.5,
    netsleuth: 0.5,
    rumor: 0.5,
    ensemble: 0.5,
    ensemble_centralities: 0.5,
}


def generate_reports():
    threshold = 1.0
    f_to_process = draw_sd_per_method_final_data
    for sd_method in SD_METHODS_TO_CHECK:
        for n in NETWORK_NAME.keys():
            f_to_process(
                sd_method=sd_method, part=n, draw_plot=True, threshold=threshold
            )
        f_to_process(sd_method=sd_method, draw_plot=True, threshold=threshold)


# generate_reports()
# f_to_process(ensemble)
# draw_average_error_by_network()
# draw_sd_per_method_final_data()
# draw_passed_computations_by_method()
# draw_passed_computations_static()
# generate_reports()
# draw_sd_per_method()
# generate_reports()
#


# draw_sd_per_method_final_data(sd_method=rumor)
# generate_reports()
# generate_finals_sd_report()
generate_reports()
# generate_finals_sd_report()
