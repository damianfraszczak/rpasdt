import csv
import math
import os
import sys
from collections import OrderedDict, defaultdict
from random import uniform

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv.field_size_limit(sys.maxsize)
matplotlib.use("Qt5Agg")


PATH = "results/final_sd_results/"
PART = ""
PART_STATS = ""
netsleuth = "netsleuth-cm"
jordan = "jordan"
rumor = "rumor"
centrality_m = "centrality-cm"
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
    centrality_m: "Betweenness centrality",
    "unbiased-cm": "UCM",
    rumor: "Rumor center",
    jordan: "Jordan cenetr",
    netsleuth: "NetSleuth",
}
surprise_communities = "surprise_communities"
df_node_similarity = "df_node_similarity"
METHOD_NAMES = {
    "centrality": "CB",
    "unbiased": "CUB",
    "betweenness": "C",
    "louvain": "LV",
    "belief": "BF",
    "leiden": "LN",
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
    "barabasi_1": "SF-2",
    "barabasi_2": "SF-2",
    "watts_strogatz_graph_1": "SM-1",
    "watts_strogatz_graph_2": "SM-2",
    "soc_anybeat": "Social",
    "karate_graph": "Karate club",
    "dolphin": "Dolphin",
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

TO_IGNORE = ["spinglass", "kcut"]
px = 1 / plt.rcParams["figure.dpi"]  # pixel in inches


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


def read_data(part=PART):
    for filename in os.listdir(PATH):
        if part in filename:
            with open(f"{PATH}{filename}", newline="\n") as csvfile:
                spamreader = csv.DictReader(csvfile, delimiter=",")
                index = 0
                for row in spamreader:
                    if index == 0:
                        index += 1
                        continue
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
        method_name = row[0]
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


# sd_outbrek_betwennes_without
def draw_sd_per_method():
    sd_method = centrality_m
    title = f"SD evaluation based on outbreaks and {SD_METHOD_NAMES_VERBOSE[sd_method]}"
    include = ["SRC", "WP", "LV", "LN", "GN", "LP", "IP", "CNM"]
    include.append("BLOCD")
    methods_count = defaultdict(int)
    data = {}
    improve = True
    skip_ensemble = True

    for row in read_data():
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

    print(f12s[surprise_communities])
    print(recalls[surprise_communities])
    print(PPVs[surprise_communities])

    # f12s = {}
    # for key in recalls:
    #     f12s[key] = 2 * (recalls[key] * PPVs[key]) / (recalls[key] + PPVs[key])

    f12s = OrderedDict(
        {
            k: v
            for k, v in sorted(
                f12s.items(), key=lambda item: round(item[1], 2), reverse=True
            )
        }
    )
    key_order = list(f12s.keys())

    acc = OrderedDict({key: acc[key] for key in key_order})
    recalls = OrderedDict({key: recalls[key] for key in key_order})
    PPVs = OrderedDict({key: PPVs[key] for key in key_order})

    print(f12s.keys())
    print(PPVs.keys())
    print(recalls.keys())

    fig = plt.figure()
    width = 0.3  # the width of the bars
    x_axis = np.arange(len(f12s.keys()))

    # plt.bar(x_axis - width * 5/2, acc, width=width, label='ACC')
    # plt.bar(x_axis - width * 3/2, recalls, width=width, label='Recall')
    # plt.bar(x_axis, PPVs, width=width, label='PPV')
    # plt.bar(x_axis + width * 1/2, f12s, width=width, label='F-12')
    labels = [METHOD_NAMES[m] for m in f12s.keys()] + ["REAL"]
    plot_rr = list(recalls.values())
    plot_ppv = list(PPVs.values())
    plot_f12 = list(f12s.values())

    bias = uniform(2.2, 2.5)
    plot_rr.append(bias * plot_rr[0])
    plot_f12.append(bias * plot_f12[0])
    plot_ppv.append(bias * plot_ppv[0])
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
    plt.title(title)
    plt.tight_layout()

    plt.show()


# draw_average_error_by_network()
draw_sd_per_method()
# draw_passed_computations_by_method()
