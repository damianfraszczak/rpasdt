import csv
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from rpasdt.algorithm.models import PropagationReconstructionConfig
from rpasdt.algorithm.propagation_reconstruction import (
    _check_if_node_is_on_path_between_infected_nodes,
    _compute_neighbors_probability,
    _init_extended_network,
    create_snapshot_IG,
    reconstruct_propagation,
)
from rpasdt.scripts.sd_samples import get_experiments
from rpasdt.scripts.taxonomies import (
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
                filename = f"results/{DIR_NAME}/regression/{graph_function.__name__}_{number_of_sources}_{delete_ratio}_regression_data.csv"
                _write_to_file(
                    header=["neighbors_probability", "node_on_path", "infected"],
                    filename=filename,
                )
                for index, experiment in enumerate(experiments):
                    G = experiment.G
                    IG = experiment.IG
                    snapshot, deleted_nodes = create_snapshot_IG(IG, delete_ratio)
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
                        _write_to_file(
                            filename=filename,
                            data=[neighbors_probability, node_on_path, int(node in IG)],
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

        return thresholds[optimal_idx], tpr[optimal_idx] - fpr[optimal_idx]

    def threshold_from_optimal_f_score(self, X, y):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        fscores = (2 * precisions * recalls) / (precisions + recalls)

        optimal_idx = np.argmax(fscores)

        return thresholds[optimal_idx], fscores[optimal_idx]

    def threshold_from_desired_precision(self, X, y, desired_precision=0.9):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        desired_precision_idx = np.argmax(precisions >= desired_precision)

        return thresholds[desired_precision_idx], recalls[desired_precision_idx]

    def threshold_from_desired_recall(self, X, y, desired_recall=0.9):
        y_scores = LogisticRegression.predict_proba(self, X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

        desired_recall_idx = np.argmin(recalls >= desired_recall)

        return thresholds[desired_recall_idx], precisions[desired_recall_idx]

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


def logistic_regression(case=""):
    # Imbalanced Data https://www.w3schools.com/python/python_ml_auc_roc.asp
    # https://towardsdatascience.com/calculating-and-setting-thresholds-to-optimise-logistic-regression-performance-c77e6d112d7e
    dir = "results/reconstruction/regression/"
    data = pd.DataFrame()
    for file in os.listdir(dir):
        if file.endswith(".csv") and case in file:
            df = pd.read_csv(dir + file)
            data = data.append(df, ignore_index=True)
    X = data[["neighbors_probability", "node_on_path"]]
    y = data["infected"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    logreg = LogisticRegressionWithThreshold(random_state=0)

    clf = logreg.fit(X_train, y_train)

    threshold, optimal_tpr_minus_fpr = logreg.threshold_from_optimal_f_score(
        X_train, y_train
    )
    print(clf.coef_, clf.intercept_, threshold)
    y_scores = logreg.predict_proba(X_test)[:, 1]
    y_pred = logreg.predict(X_test)

    auc = metrics.roc_auc_score(y_test, y_pred)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print("AUC:", auc)

    fpr, tpr, thresholds = roc_curve(y_test, y_scores)
    # create ROC curve

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.title("ROC Curve " + case)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend(loc=4)
    plt.show()


def evaluation():
    for graph in graphs:
        logistic_regression(graph.__name__)


# generate_data_for_regression(True)
# evaluation()

logistic_regression()
