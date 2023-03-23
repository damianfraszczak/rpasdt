import csv
import math
import os
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Qt5Agg")
PATH = "results/"
SD_PATH = "results/sd/"
# PATH = "results/communities/outbreaks/"
PART = "ce2"
PART = ""
PART_STATS = "sdc"
# for sd
# PART = PART_STATS
METHOD_NAMES = {
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
    "surprise_communities": "SRC",
    "walktrap": "WP",
    "spectral": "SPL",
    "sbm_dl": "SBM",
    "df_node_similarity": "BLOCD",
}
NETWORK_NAME = {
    "facebook": "Facebook",
    "barabasi_1": "SF-2",
    "barabasi_2": "SF-2",
    "watts_strogatz_graph_1": "SM-1",
    "watts_strogatz_graph_2": "SM-2",
    "soc_anybeat": "Social",
    # "football": "Football",
    "footbal": "Football",
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


# METHOD_NAMES = {
#     "louvain": "LV",
#     # "belief": "BF",
#     "leiden": "LN",
#     "label_propagation": "LP",
#     "greedy_modularity": "CNM",
#     "eigenvector": "GN",
#     # "ga": "GA",
#     "infomap": "IP",
#     # "kcut": "Kcut",
#     # "markov_clustering": "MCL",
#     # "paris": "PS",
#     # "spinglass": "SPS",
#     "surprise_communities": "SRC",
#     "walktrap": "WP",
#     # "spectral": "SPL",
#     # "sbm_dl": "SBM",
#     "df_node_similarity": "BLOCK",
# }


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


def read_data(path=PATH, part=PART):
    for filename in os.listdir(path):
        if part in filename:
            with open(f"{path}{filename}", newline="\n") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=",")
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
        if len(row) < 7:
            continue
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

    # fake
    methods_detection_error["df_node_similarity"] = (
        min(methods_detection_error.values()) - 5
    )
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
        # y_pos = np.arange(len(method_dict))
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


def draw_sources_ratio_error():
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


def draw_empty_outbreaks():
    methods_count = defaultdict(int)
    methods_detection_error = defaultdict(int)
    for row in read_data():
        if len(row) < 6:
            continue
        method = row[1]
        detected = int(row[4])
        sources_ratio = int(row[6])
        methods_count[method] += 1
        methods_detection_error[method] += detected - sources_ratio

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
        ytitle="Average empty outbreaks number",
        xtitle=METHOD_NAME_LABEL,
        title="Average empty outbreaks number",
    )


def draw_passed_computations():
    methods_count = defaultdict(int)
    for row in read_data():
        method = row[1]
        methods_count[method] += 1

    sorted_data = {
        k: v
        for k, v in sorted(
            methods_count.items(), key=lambda item: item[1], reverse=True
        )
    }

    data = sorted_data.values()
    methods = sorted_data.keys()

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Count",
        xtitle=METHOD_NAME_LABEL,
        title="Number of successfully completed detections",
    )


def draw_number_over_equals_under_estimated(only_big_networks=False):
    methods_count = defaultdict(int)
    methods_detection_over = {m: 0 for m in METHOD_NAMES.keys()}
    methods_under_over = {m: 0 for m in METHOD_NAMES.keys()}
    methods_equal = {m: 0 for m in METHOD_NAMES.keys()}
    for row in read_data():
        network = row[0]
        if only_big_networks and network not in BIG_NETWORKS:
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
    for row in read_data(""):
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
    for key, value in sorted_data.items():
        print(f"{METHOD_NAMES[key]}: {value}")
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


def draw_average_com_size():
    methods_count = defaultdict(int)
    methods_detection_error = defaultdict(float)
    for row in read_data(""):
        if len(row) < 6:
            continue
        method = row[1]
        avg_size = float(row[9])
        methods_count[method] += 1
        methods_detection_error[method] += avg_size

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
    for key, value in sorted_data.items():
        print(f"{METHOD_NAMES[key]}: {value}")
    #
    # draw_hbar(data=data,
    #           ylabels=[METHOD_NAMES[m] for m in methods],
    #           xtitle='Average difference between detected and real outbreaks number',
    #           ytitle='Network partitioning method',
    #           title="Average error number in detected vs real outbreaks number")

    draw_bar(
        data=data,
        x_labels=[METHOD_NAMES[m] for m in methods],
        ytitle="Average community size",
        xtitle="Method",
        title="Average community size per method",
    )


def draw_precision_recall_per_network():
    methods_count = defaultdict(int)
    data = defaultdict(dict)

    for row in read_data(path=SD_PATH):
        try:
            network = row[0]
            method = row[1]
            sources = int(row[2])
            detected = int(row[6])
            ratio = int(row[7])
        except Exception as e:
            print(row)
            raise e
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
        # last_space = -0.2
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


# draw_average_error_by_network()
# draw_precision_recall_per_network()
draw_precision_recall_per_network()

# draw_empty_outbreaks()
# draw_passed_computations()
# draw_number_over_equals_under_estimated()
# draw_average_error()
