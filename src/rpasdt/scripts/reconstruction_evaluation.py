import csv
import os

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import argmax, sqrt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from rpasdt.algorithm.models import PropagationReconstructionConfig
from rpasdt.algorithm.plots import _configure_plot
from rpasdt.algorithm.propagation_reconstruction import (
    _check_if_node_is_on_path_between_infected_nodes,
    _compute_neighbors_probability,
    _init_extended_network,
    create_snapshot_IG,
    reconstruct_propagation,
)
from rpasdt.scripts.sd_samples import get_experiments
from rpasdt.scripts.taxonomies import (
    NETWORK_NAMES,
    barabasi_1,
    barabasi_2,
    dolphin,
    facebook,
    footbal,
    karate_graph,
    soc_anybeat,
    watts_strogatz_graph_1,
    watts_strogatz_graph_2,
)

DIR_NAME = "reconstruction"
graphs = [
    karate_graph,
    dolphin,
    footbal,
    barabasi_1,
    barabasi_2,
    watts_strogatz_graph_1,
    watts_strogatz_graph_2,
    facebook,
    soc_anybeat,
]

deleted_ratios = [5, 10, 15, 20, 25, 30]

grid_search = [
    (0.05, 0.95),
    (0.1, 0.9),
    (0.15, 0.85),
    (0.2, 0.8),
    (0.25, 0.75),
    (0.3, 0.7),
    (0.35, 0.65),
    (0.4, 0.6),
    (0.45, 0.55),
    (0.5, 0.5),
    (0.55, 0.45),
    (0.6, 0.4),
    (0.65, 0.35),
    (0.7, 0.3),
    (0.75, 0.25),
    (0.8, 0.2),
    (0.85, 0.15),
    (0.9, 0.1),
    (0.95, 0.05),
]
thresholds = [
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

header = [
    "number_of_sources",
    "index",
    "delete_ratio",
    "node",
    "neighbors_probability",
    "node_on_path",
    "infected",
    "sample_decision",
    "correct",
]
header_evaluate = [
    "index",
    "number_of_sources",
    "delete_ratio",
    "deleted_nodes",
    "reconstructed_nodes",
    "m1",
    "m2",
    "threshold",
    "ALL",
    "TP",
    "TN",
    "FP",
    "FN",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


def serialize_nodes(nodes):
    return "|".join([str(node) for node in sorted(nodes)])


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


def generate_data(only_biggest=False):
    for graph_function in graphs:
        experiments = get_experiments(graph_function)
        filename = f"results/{DIR_NAME}/{graph_function.__name__}.csv"
        _write_to_file(header=header, filename=filename)
        correct_count = 0
        incorrect_count = 0
        max_number_of_sources = max(experiments.keys())
        for number_of_sources, experiments in experiments.items():
            if only_biggest and number_of_sources != max_number_of_sources:
                continue
            for index, experiment in enumerate(experiments):
                for delete_ratio in deleted_ratios:
                    G = experiment.G
                    IG = experiment.IG
                    snapshot, _ = create_snapshot_IG(IG, delete_ratio)
                    EG = _init_extended_network(G=G, IG=snapshot)
                    for node in EG:
                        if node in snapshot:
                            continue
                        neighbors_probability = _compute_neighbors_probability(
                            node=node, G=EG
                        )
                        node_on_path = int(
                            _check_if_node_is_on_path_between_infected_nodes(
                                node=node, G=EG
                            )
                        )
                        sample_result = 0.5 * neighbors_probability + 0.5 * node_on_path
                        decision = sample_result >= 0.8
                        correct = decision == (node in IG)
                        if correct:
                            correct_count += 1
                        else:
                            incorrect_count += 1
                        _write_to_file(
                            filename=filename,
                            data=[
                                number_of_sources,
                                index,
                                delete_ratio,
                                node,
                                neighbors_probability,
                                node_on_path,
                                node in IG,
                                decision,
                                correct,
                            ],
                        )
        print(f"{filename}, correct: {correct_count}, incorrect: {incorrect_count}")


def evaluate_data():
    for graph_function in graphs:
        experiments = get_experiments(graph_function)
        filename = f"results/{DIR_NAME}/{graph_function.__name__}_evaluation.csv"
        _write_to_file(header=header_evaluate, filename=filename)
        for number_of_sources, experiments in experiments.items():

            for index, experiment in enumerate(experiments):
                for delete_ratio in deleted_ratios:
                    G = experiment.G
                    IG = experiment.IG
                    snapshot, deleted_nodes = create_snapshot_IG(IG, delete_ratio)
                    for m1, m2 in grid_search:
                        for threshold in thresholds:
                            EG = reconstruct_propagation(
                                PropagationReconstructionConfig(
                                    G=G,
                                    IG=snapshot,
                                    real_IG=IG,
                                    m1=m1,
                                    m2=m2,
                                    threshold=threshold,
                                    max_iterations=1,
                                )
                            )
                            not_infected = [node for node in G if node not in IG]
                            reconstructed_nodes = [
                                node for node in EG if node not in snapshot
                            ]
                            TP = len(
                                [
                                    node
                                    for node in reconstructed_nodes
                                    if node in deleted_nodes
                                ]
                            )
                            TN = len(
                                [
                                    node
                                    for node in not_infected
                                    if node not in reconstructed_nodes
                                ]
                            )
                            FP = len(
                                [
                                    node
                                    for node in reconstructed_nodes
                                    if node not in deleted_nodes
                                ]
                            )
                            FN = len(
                                [
                                    node
                                    for node in deleted_nodes
                                    if node in reconstructed_nodes
                                ]
                            )
                            accuracy = (TP + TN) / max((TP + TN + FP + FN), 1)
                            precision = TP / max((TP + FP), 1)
                            recall = TP / max((TP + FN), 1)
                            f1 = 2 * (precision * recall) / max((precision + recall), 1)

                            _write_to_file(
                                filename=filename,
                                data=[
                                    index,
                                    number_of_sources,
                                    delete_ratio,
                                    serialize_nodes(deleted_nodes),
                                    serialize_nodes(reconstructed_nodes),
                                    m1,
                                    m2,
                                    threshold,
                                    len(reconstructed_nodes),
                                    TP,
                                    TN,
                                    FP,
                                    FN,
                                    accuracy,
                                    precision,
                                    recall,
                                    f1,
                                ],
                            )


def generate_data_for_regression(only_biggest=False):
    for graph_function in graphs:
        loaded_experiments = get_experiments(graph_function)
        max_number_of_sources = max(loaded_experiments.keys())
        for delete_ratio in deleted_ratios:
            for number_of_sources, experiments in loaded_experiments.items():
                if only_biggest and number_of_sources != max_number_of_sources:
                    continue
                ratio_of_sources = int(number_of_sources / len(graph_function()) * 100)
                filename = f"results/{DIR_NAME}/regression/{graph_function.__name__}_ratio:{ratio_of_sources}_deleted:{delete_ratio}_regression_data.csv"
                _write_to_file(
                    header=[
                        "neighbors_probability",
                        "node_on_path",
                        "degree",
                        "infected",
                    ],
                    filename=filename,
                )
                for index, experiment in enumerate(experiments):
                    G = experiment.G
                    IG = experiment.IG
                    snapshot, deleted_nodes = create_snapshot_IG(IG, delete_ratio)
                    EG = _init_extended_network(G=G, IG=snapshot)
                    degree = nx.degree_centrality(EG)
                    for node in EG:
                        if node in snapshot:
                            continue
                        neighbors_probability = _compute_neighbors_probability(
                            node=node, G=EG
                        )
                        node_on_path = _check_if_node_is_on_path_between_infected_nodes(
                            node=node, G=EG
                        )
                        node_degree = degree[node]
                        node_infected = int(node in IG)
                        _write_to_file(
                            filename=filename,
                            data=[
                                neighbors_probability,
                                node_on_path,
                                node_degree,
                                node_infected,
                            ],
                        )


class LogisticRegressionWithThreshold(LogisticRegression):
    def predict(self, X, threshold=None):
        if (
            threshold is None
        ):  # If no threshold passed in, simply call the base class predict, effectively threshold=0.5
            return LogisticRegression.predict(self, X)
        else:
            y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
            y_pred_with_threshold = (y_scores >= threshold).astype(int)

            return y_pred_with_threshold

    def threshold_from_optimal_tpr_minus_fpr(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, y_scores)

        optimal_idx = np.argmax(tpr - fpr)

        return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx], optimal_idx

    def threshold_from_optimal_f_score(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        fscores = (2 * precisions * recalls) / (precisions + recalls)

        optimal_idx = np.argmax(fscores)

        return thresholds[optimal_idx], fscores[optimal_idx], optimal_idx

    def threshold_from_optimal_recall(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        optimal_idx = np.argmax(recalls)

        return thresholds[optimal_idx], recalls[optimal_idx], optimal_idx

    def threshold_from_desired_precision(self, X, y, desired_precision=0.9):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        desired_precision_idx = np.argmax(precisions >= desired_precision)

        return (
            thresholds[desired_precision_idx],
            recalls[desired_precision_idx],
            desired_precision_idx,
        )

    def threshold_from_desired_recall(self, X, y, desired_recall=0.9):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        desired_recall_idx = np.argmin(recalls >= desired_recall)

        return (
            thresholds[desired_recall_idx],
            precisions[desired_recall_idx],
            desired_recall_idx,
        )

    def threshold_from_cost_function(self, X, y, cost_function):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        costs = []
        tns = []
        fps = []
        fns = []
        tps = []

        for threshold in thresholds:
            y_pred_with_threshold = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_pred_with_threshold).ravel()
            costs.append(cost_function(tn, fp, fn, tp))
            tns.append(tn), fps.append(fp), fns.append(fn), tps.append(tp)

        df_cost = pd.DataFrame(
            {
                "precision": precisions[:-1],
                "recall": recalls[:-1],
                "threshold": thresholds,
                "cost": costs,
                "tn": tns,
                "fp": fps,
                "fn": fns,
                "tp": tps,
            }
        )

        min_cost = df_cost["cost"].min()
        threshold = df_cost[df_cost["cost"] == min_cost].iloc[0]["threshold"]

        return threshold, min_cost, df_cost

    def threshold_from_optimal_accuracy(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        accuracies = []
        tns = []
        fps = []
        fns = []
        tps = []

        for threshold in thresholds:
            y_pred_with_threshold = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, y_pred_with_threshold).ravel()

            accuracies.append((tp + tn) / (tn + fp + fn + tp))
            tns.append(tn), fps.append(fp), fns.append(fn), tps.append(tp)

        df_accuracy = pd.DataFrame(
            {
                "threshold": thresholds,
                "accuracy": accuracies,
                "tn": tns,
                "fp": fps,
                "fn": fns,
                "tp": tps,
            }
        )

        max_accuracy = df_accuracy["accuracy"].max()
        threshold = df_accuracy[df_accuracy["accuracy"] == max_accuracy].iloc[0][
            "threshold"
        ]

        return threshold, max_accuracy, df_accuracy


def threshold_from_optimal_tpr_minus_fpr(y, y_scores):
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    optimal_idx = np.argmax(tpr - fpr)
    return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx]


def threshold_from_optimal_f_score(y, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)
    fscores = (2 * precisions * recalls) / (precisions + recalls)
    optimal_idx = np.argmax(fscores)
    return thresholds[optimal_idx], fscores[optimal_idx]


def grid_search_function(case=""):
    dir = "results/reconstruction/regression/"
    data = pd.DataFrame()
    for file in os.listdir(dir):
        if file.endswith(".csv") and case in file:
            df = pd.read_csv(dir + file)
            data = data.append(df, ignore_index=True)
    y_test = data["infected"]
    best_auc, best_threshold = -1, -1
    best_m1, best_m2, best_ypred, best_fpr, best_tpr = 0, 0, [], [], []
    for m1, m2 in grid_search:
        y_scores = data["neighbors_probability"] * m1 + data["node_on_path"] * m2
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)

        best_threshold = threshold_from_optimal_f_score(y_test, y_scores)
        y_pred = [x >= best_threshold for x in y_scores]

        auc = metrics.roc_auc_score(y_test, y_pred)
        if auc > best_auc:
            best_auc = auc
            best_m1 = m1
            best_m2 = m2
            best_ypred = y_pred
            best_fpr = fpr
            best_tpr = tpr

    print("Best m1,m2", (best_m1, best_m2))
    print("Best threshold", best_threshold)

    print("Accuracy:", metrics.accuracy_score(y_test, best_ypred))
    print("Precision:", metrics.precision_score(y_test, best_ypred))
    print("Recall:", metrics.recall_score(y_test, best_ypred))
    print("F1:", metrics.f1_score(y_test, best_ypred))
    plt.plot(best_fpr, best_tpr, label="AUC=" + str(best_auc))
    plt.title("ROC Curve " + case)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()


def _read_data(dir, case=""):
    data = pd.DataFrame()
    for file in os.listdir(dir):
        if file.endswith(".csv") and case in file:
            df = pd.read_csv(dir + file)
            data = data.append(df, ignore_index=True)
    return data


def _plot_scatter_for_data(network, data, x1, x2, y):
    infected = data[data["infected"] == 1]
    not_infected = data[data["infected"] == 0]
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(x=infected[x1], y=infected[x2], label="Zainfekowany", marker="x")
    ax2.scatter(
        x=not_infected[x1],
        y=not_infected[x2],
        label="Niezainfekowany",
        marker="o",
        alpha=0.05,
    )

    _configure_plot(
        title=f"Dane uczące sieci {network}", ylabel="RIN_d(x)", xlabel="IB_d(x)"
    )

    plt.show()


def _draw_scatter_plots():
    dir = "results/reconstruction/regression/"
    for graph in graphs:
        case = graph.__name__
        data = _read_data(dir, case=case)

        # print(data["infected"].value_counts())
        # rozklad
        print(data["infected"].value_counts())
        print(data["infected"].value_counts() / data.shape[0])

        data["neighbors_probability"] = data["neighbors_probability"] * data["degree"]
        data["node_on_path"] = data["node_on_path"] * data["degree"]
        _plot_scatter_for_data(
            case, data, "neighbors_probability", "node_on_path", "infected"
        )


def _format_number(val, rounds=3):
    return f"{round(val, rounds)}"


def logistic_regression_with_roc(case=""):
    dir = "results/reconstruction/regression/"
    data = _read_data(dir, case)

    # print(data["infected"].value_counts())
    # rozklad
    print(case)
    print(data["infected"].value_counts())
    print(data["infected"].value_counts() / data.shape[0])

    data["neighbors_probability"] = data["neighbors_probability"] * data["degree"]
    data["node_on_path"] = data["node_on_path"] * data["degree"]
    X = data[["neighbors_probability", "node_on_path"]]
    y = data["infected"]
    # scatter plot
    # _plot_scatter_for_data(case, data, "neighbors_probability", "node_on_path", "infected")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # by znalezsc najlepszy class weight https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b
    # define weight hyperparameter
    # w = [
    #     {0: 1000, 1: 100},
    #     {0: 1000, 1: 10},
    #     {0: 1000, 1: 1.0},
    #     {0: 500, 1: 1.0},
    #     {0: 400, 1: 1.0},
    #     {0: 300, 1: 1.0},
    #     {0: 200, 1: 1.0},
    #     {0: 150, 1: 1.0},
    #     {0: 100, 1: 1.0},
    #     {0: 99, 1: 1.0},
    #     {0: 10, 1: 1.0},
    #     {0: 0.01, 1: 1.0},
    #     {0: 0.01, 1: 10},
    #     {0: 0.01, 1: 100},
    #     {0: 0.001, 1: 1.0},
    #     {0: 0.005, 1: 1.0},
    #     {0: 1.0, 1: 1.0},
    #     {0: 1.0, 1: 0.1},
    #     {0: 10, 1: 0.1},
    #     {0: 100, 1: 0.1},
    #     {0: 10, 1: 0.01},
    #     {0: 1.0, 1: 0.01},
    #     {0: 1.0, 1: 0.001},
    #     {0: 1.0, 1: 0.005},
    #     {0: 1.0, 1: 10},
    #     {0: 1.0, 1: 99},
    #     {0: 1.0, 1: 100},
    #     {0: 1.0, 1: 150},
    #     {0: 1.0, 1: 200},
    #     {0: 1.0, 1: 300},
    #     {0: 1.0, 1: 400},
    #     {0: 1.0, 1: 500},
    #     {0: 1.0, 1: 1000},
    #     {0: 10, 1: 1000},
    #     {0: 100, 1: 1000},
    # ]
    # hyperparam_grid = {"class_weight": w}
    # {'class_weight': {0: 500, 1: 1.0}}
    logreg = LogisticRegressionWithThreshold(random_state=0, class_weight="balanced")
    # grid = GridSearchCV(logreg, hyperparam_grid, scoring="roc_auc", cv=100,
    #                     n_jobs=-1, refit=True)
    # grid.fit(X_train, y_train)
    # print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

    logreg.fit(X_train, y_train)
    y_scores = logreg.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    threshold, optimal_val, optimal_id = logreg.threshold_from_optimal_tpr_minus_fpr(
        X_test, y_test
    )
    optimal_id = np.argmax(tpr - fpr)
    gmeans = sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    threshold2 = thresholds[ix]
    y_pred = logreg.predict(X_test, threshold)
    y_pred2 = logreg.predict(X_test, threshold2)

    # create ROC curve
    auc = metrics.roc_auc_score(y_test, y_pred)
    auc2 = metrics.roc_auc_score(y_test, y_pred2)

    print("CASE", case)
    print(
        f"Accuracy: ({metrics.accuracy_score(y_test, y_pred)}) vs ({metrics.accuracy_score(y_test, y_pred2)})",
    )
    print(
        f"Precision: ({metrics.precision_score(y_test, y_pred)}) vs ({metrics.precision_score(y_test, y_pred2)})"
    )
    print(
        f"Recall: ({metrics.recall_score(y_test, y_pred)}) vs ({metrics.recall_score(y_test, y_pred2)})"
    )
    print(
        f"F1:  ({metrics.f1_score(y_test, y_pred)}) vs ({metrics.f1_score(y_test, y_pred2)})"
    )
    print(f"AUC: ({auc}) vs ({auc2})")

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    ax1.plot(fpr, tpr, label="AUC=" + str(auc))
    ax1.scatter(
        fpr[optimal_id], tpr[optimal_id], marker="o", color="black", label="Best"
    )
    ax1.set_title("ROC Curve TPR" + case + " threshold  " + str(threshold))
    ax1.set_ylabel("True Positive Rate")
    ax1.set_xlabel("False Positive Rate")
    ax1.legend(loc=4)

    ax2.plot(fpr, tpr, label="AUC=" + str(auc2))
    ax2.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    ax2.scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    ax2.set_title("ROC Curve RECALL " + case + " threshold  " + str(threshold2))
    ax2.set_ylabel("True Positive Rate")
    ax2.set_xlabel("False Positive Rate")

    plt.legend(loc=4)
    plt.tight_layout()
    plt.show()


CASE_TITLES = {"": "ogólnie", **NETWORK_NAMES}


def logistic_regression_with_roc_and_pr(case=""):
    dir = "results/reconstruction/regression/"
    data = _read_data(dir, case)
    case_title = f"dla sieci {CASE_TITLES[case]}" if case else "ogólnie"
    # print(data["infected"].value_counts())
    # rozklad
    print(case)
    print(data["infected"].value_counts())
    print(data["infected"].value_counts() / data.shape[0])

    data["neighbors_probability"] = data["neighbors_probability"] * data["degree"]
    data["node_on_path"] = data["node_on_path"] * data["degree"]
    X = data[["neighbors_probability", "node_on_path"]]
    y = data["infected"]
    # scatter plot
    # _plot_scatter_for_data(case, data, "neighbors_probability", "node_on_path", "infected")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # by znalezsc najlepszy class weight https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b
    # define weight hyperparameter
    # w = [
    #     {0: 1000, 1: 100},
    #     {0: 1000, 1: 10},
    #     {0: 1000, 1: 1.0},
    #     {0: 500, 1: 1.0},
    #     {0: 400, 1: 1.0},
    #     {0: 300, 1: 1.0},
    #     {0: 200, 1: 1.0},
    #     {0: 150, 1: 1.0},
    #     {0: 100, 1: 1.0},
    #     {0: 99, 1: 1.0},
    #     {0: 10, 1: 1.0},
    #     {0: 0.01, 1: 1.0},
    #     {0: 0.01, 1: 10},
    #     {0: 0.01, 1: 100},
    #     {0: 0.001, 1: 1.0},
    #     {0: 0.005, 1: 1.0},
    #     {0: 1.0, 1: 1.0},
    #     {0: 1.0, 1: 0.1},
    #     {0: 10, 1: 0.1},
    #     {0: 100, 1: 0.1},
    #     {0: 10, 1: 0.01},
    #     {0: 1.0, 1: 0.01},
    #     {0: 1.0, 1: 0.001},
    #     {0: 1.0, 1: 0.005},
    #     {0: 1.0, 1: 10},
    #     {0: 1.0, 1: 99},
    #     {0: 1.0, 1: 100},
    #     {0: 1.0, 1: 150},
    #     {0: 1.0, 1: 200},
    #     {0: 1.0, 1: 300},
    #     {0: 1.0, 1: 400},
    #     {0: 1.0, 1: 500},
    #     {0: 1.0, 1: 1000},
    #     {0: 10, 1: 1000},
    #     {0: 100, 1: 1000},
    # ]
    # hyperparam_grid = {"class_weight": w}
    # {'class_weight': {0: 500, 1: 1.0}}
    logreg = LogisticRegressionWithThreshold(random_state=0, class_weight="balanced")
    # grid = GridSearchCV(logreg, hyperparam_grid, scoring="roc_auc", cv=100,
    #                     n_jobs=-1, refit=True)
    # grid.fit(X_train, y_train)
    # print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

    logreg.fit(X_train, y_train)
    y_scores = logreg.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    threshold, optimal_val, optimal_id = logreg.threshold_from_optimal_tpr_minus_fpr(
        X_test, y_test
    )
    optimal_id = np.argmax(tpr - fpr)
    gmeans = sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    threshold2 = thresholds[ix]

    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
    # convert to f score
    pr_threshold, pre, ix_rc = logreg.threshold_from_optimal_f_score(X_test, y_test)

    y_pred = logreg.predict(X_test, threshold)
    y_pred2 = logreg.predict(X_test, threshold2)
    y_pred3 = logreg.predict(X_test, pr_threshold)

    # create ROC curve
    auc = metrics.roc_auc_score(y_test, y_pred)
    auc2 = metrics.roc_auc_score(y_test, y_pred2)
    auc3 = metrics.auc(recall, precision)

    print("CASE", case)
    print(
        f"Accuracy: ({metrics.accuracy_score(y_test, y_pred)}) vs ({metrics.accuracy_score(y_test, y_pred3)})",
    )
    print(
        f"Precision: ({metrics.precision_score(y_test, y_pred)}) vs ({metrics.precision_score(y_test, y_pred3)})"
    )
    print(
        f"Recall: ({metrics.recall_score(y_test, y_pred)}) vs ({metrics.recall_score(y_test, y_pred3)})"
    )
    print(
        f"F1:  ({metrics.f1_score(y_test, y_pred)}) vs ({metrics.f1_score(y_test, y_pred3)})"
    )
    print(f"AUC: ({auc}) vs ({auc3})")

    fig, (ax1, ax2) = plt.subplots(2)

    ax1.plot([0, 1], [0, 1], linestyle="--", label="Losowy")
    ax1.plot(fpr, tpr, label="AUC=" + _format_number(auc))
    ax1.scatter(
        fpr[optimal_id],
        tpr[optimal_id],
        marker="o",
        color="black",
        label="Optymalny threshold",
    )
    ax1.set_title(
        "Krzywa ROC " + case_title + ", threshold  " + _format_number(threshold)
    )
    ax1.set_ylabel("TPR")
    ax1.set_xlabel("FPR")
    ax1.legend(loc=4)

    # ax2.plot(fpr, tpr, label="AUC=" + str(auc2))
    # ax2.plot([0, 1], [0, 1], linestyle="--", label="No Skill")
    # ax2.scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    # ax2.set_title("ROC Curve RECALL " + case + " threshold  " + str(threshold2))
    # ax2.set_ylabel("True Positive Rate")
    # ax2.set_xlabel("False Positive Rate")
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    ax2.plot([0, 1], [no_skill, no_skill], linestyle="--", label="Losowy")
    ax2.plot(recall, precision, marker=".", label="AUC=" + _format_number(auc3))
    ax2.scatter(
        recall[ix_rc],
        precision[ix_rc],
        marker="o",
        color="black",
        label="Optymalny threshold",
    )
    # axis labels
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precyzja")
    ax2.set_title(
        "Krzywa precyzji/czułość "
        + case_title
        + ", threshold  "
        + _format_number(pr_threshold)
    )
    ax2.legend(loc=1)
    # plt.legend(loc=4)
    plt.tight_layout()
    plt.show()


def logistic_regression_with_recall(case=""):
    dir = "results/reconstruction/regression/"
    data = _read_data(dir, case)

    # print(data["infected"].value_counts())
    # rozklad
    print(data["infected"].value_counts() / data.shape[0])
    data["neighbors_probability"] = data["neighbors_probability"] * data["degree"]
    data["node_on_path"] = data["node_on_path"] * data["degree"]
    X = data[["neighbors_probability", "node_on_path"]]
    y = data["infected"]
    # scatter plot
    # _plot_scatter_for_data(data, "neighbors_probability", "node_on_path", "infected")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    logreg = LogisticRegressionWithThreshold(random_state=0)
    logreg.fit(X_train, y_train)

    y_scores = logreg.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    # convert to f score
    threshold, pre, ix = logreg.threshold_from_optimal_f_score(X_test, y_test)
    # locate the index of the largest f score

    y_pred = logreg.predict(X_test, threshold)
    auc = metrics.auc(recall, precision)

    print("CASE", case)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("AUC:", auc)

    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    plt.plot(recall, precision, marker=".", label="AUC=" + str(auc))
    plt.scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")
    # axis labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve " + case + " threshold  " + str(threshold))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    y_pred = logreg.predict(X_test, threshold)


def logistic_regression(case=""):
    # Imbalanced Data https://www.w3schools.com/python/python_ml_auc_roc.asp
    # https://towardsdatascience.com/calculating-and-setting-thresholds-to-optimise-logistic-regression-performance-c77e6d112d7e
    # dorobic to https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    dir = "results/reconstruction/regression/"
    data = _read_data(dir, case)

    # print(data["infected"].value_counts())
    # rozklad
    print(data["infected"].value_counts() / data.shape[0])

    X = data[["neighbors_probability", "node_on_path", "degree"]]
    y = data["infected"]
    # scatter plot
    # _plot_scatter_for_data(data, "neighbors_probability", "node_on_path", "infected")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # by znalezsc najlepszy class weight https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b
    # define weight hyperparameter
    # w = [
    #     {0: 1000, 1: 100},
    #     {0: 1000, 1: 10},
    #     {0: 1000, 1: 1.0},
    #     {0: 500, 1: 1.0},
    #     {0: 400, 1: 1.0},
    #     {0: 300, 1: 1.0},
    #     {0: 200, 1: 1.0},
    #     {0: 150, 1: 1.0},
    #     {0: 100, 1: 1.0},
    #     {0: 99, 1: 1.0},
    #     {0: 10, 1: 1.0},
    #     {0: 0.01, 1: 1.0},
    #     {0: 0.01, 1: 10},
    #     {0: 0.01, 1: 100},
    #     {0: 0.001, 1: 1.0},
    #     {0: 0.005, 1: 1.0},
    #     {0: 1.0, 1: 1.0},
    #     {0: 1.0, 1: 0.1},
    #     {0: 10, 1: 0.1},
    #     {0: 100, 1: 0.1},
    #     {0: 10, 1: 0.01},
    #     {0: 1.0, 1: 0.01},
    #     {0: 1.0, 1: 0.001},
    #     {0: 1.0, 1: 0.005},
    #     {0: 1.0, 1: 10},
    #     {0: 1.0, 1: 99},
    #     {0: 1.0, 1: 100},
    #     {0: 1.0, 1: 150},
    #     {0: 1.0, 1: 200},
    #     {0: 1.0, 1: 300},
    #     {0: 1.0, 1: 400},
    #     {0: 1.0, 1: 500},
    #     {0: 1.0, 1: 1000},
    #     {0: 10, 1: 1000},
    #     {0: 100, 1: 1000},
    # ]
    # hyperparam_grid = {"class_weight": w}
    # {'class_weight': {0: 500, 1: 1.0}}
    logreg = LogisticRegressionWithThreshold(
        solver="lbfgs", random_state=0, class_weight="balanced"
    )
    # grid = GridSearchCV(logreg, hyperparam_grid, scoring="roc_auc", cv=100,
    #                     n_jobs=-1, refit=True)
    # grid.fit(X_train, y_train)
    # print(f'Best score: {grid.best_score_} with param: {grid.best_params_}')

    logreg.fit(X_train, y_train)
    y_scores = logreg.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    lr_precision, lr_recall, threshold_recall = precision_recall_curve(y_test, y_scores)

    threshold, optimal_val, optimal_id = logreg.threshold_from_optimal_tpr_minus_fpr(
        X_train, y_train
    )

    y_pred = logreg.predict(X_test, threshold)

    # create ROC curve

    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.scatter(
        fpr[optimal_id], tpr[optimal_id], marker="o", color="black", label="Best"
    )
    plt.title("ROC Curve " + case + " threshold  " + str(threshold))
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()
    # calculate precision and recall for each threshold

    # calculate scores
    lr_f1, lr_auc = f1_score(y_test, y_pred), metrics.auc(lr_recall, lr_precision)
    # convert to f score
    fscore = (2 * lr_precision * lr_recall) / (lr_precision + lr_recall)
    # locate the index of the largest f score
    ix = argmax(fscore)
    # summarize scores
    print("Logistic: f1=%.3f auc=%.3f" % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    plt.plot(lr_recall, lr_precision, marker=".", label="Logistic")
    plt.scatter(
        lr_recall[ix], lr_precision[ix], marker="o", color="black", label="Best"
    )
    # axis labels
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve " + case + " threshold  " + str(threshold))
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    auc = metrics.auc(lr_recall, lr_precision)

    print("CASE", case)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("AUC:", auc)


def evaluation(skip_graphs=False):
    if not skip_graphs:
        for graph in graphs:
            logistic_regression_with_roc_and_pr(graph.__name__)
    # logistic_regression_with_roc_and_pr()


# generate_data_for_regression(False)
# evaluation()


def evaluation_grid():
    for graph in graphs:
        grid_search_function(graph.__name__)


evaluation()
# evaluation_grid()

# generate_data_for_regression()
# _draw_scatter_plots()
