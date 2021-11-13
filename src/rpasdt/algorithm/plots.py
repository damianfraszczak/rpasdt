"""Plots utilities."""
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from ndlib.models.DiffusionModel import DiffusionModel
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from sklearn.metrics import ConfusionMatrixDisplay

from rpasdt.algorithm.models import ClassificationMetrics

matplotlib.use("Qt5Agg")

AXIS_FONT_SIZE = 14
LEGEND_FONT_SIZE = 10


def _configure_plot(
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
):
    title = title or plt.gca().get_title()
    ylabel = ylabel or plt.gca().get_ylabel()
    xlabel = xlabel or plt.gca().get_xlabel()

    plt.title(title, fontsize=AXIS_FONT_SIZE)
    plt.xlabel(xlabel, fontsize=AXIS_FONT_SIZE)
    plt.ylabel(ylabel, fontsize=AXIS_FONT_SIZE)
    plt.legend(loc="best", fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()


class NdlibDiffusionPlotMixin:
    def initialize_parent(self, trends, available_statuses, nodes_count):
        self.trends = trends
        statuses = available_statuses
        self.srev = {v: k for k, v in statuses.items()}
        self.ylabel = ""
        self.title = ""
        self.nnodes = nodes_count
        self.normalized = True
        setattr(self, "plot", self.custom_plot)

    def custom_plot(self, filename=None, percentile=90, statuses=None):
        """
        Generate the plot.

        :param filename: Output filename
        :param percentile: The percentile for the trend variance area
        :param statuses: List of statuses to plot. If not specified all statuses trends will be shown.
        """
        matplotlib.use("Qt5Agg")
        pres = self.iteration_series(percentile)
        mx = 0
        i = 0
        for k, l in pres.items():

            if statuses is not None and self.srev[k] not in statuses:
                continue
            mx = len(l[0])
            if self.normalized:
                plt.plot(
                    list(range(0, mx)),
                    l[1] / self.nnodes,
                    lw=2,
                    label=self.srev[k],
                    alpha=0.5,
                )  # , color=cols[i])
                plt.fill_between(
                    list(range(0, mx)),
                    l[0] / self.nnodes,
                    l[2] / self.nnodes,
                    alpha=0.2,
                )
                # ,color=cols[i])
            else:
                plt.plot(
                    list(range(0, mx)), l[1], lw=2, label=self.srev[k], alpha=0.5
                )  # , color=cols[i])
                plt.fill_between(
                    list(range(0, mx)), l[0], l[2], alpha=0.2
                )  # ,color=cols[i])

            i += 1
        plt.xlim((0, mx))
        plt.grid(axis="y")
        _configure_plot(title=self.title, xlabel="Iterations", ylabel=self.ylabel)

        if filename is not None:
            plt.savefig(filename)
            plt.clf()


class DiffusionTrendPlot(DiffusionTrend, NdlibDiffusionPlotMixin):
    def __init__(
        self,
        trends,
        available_statuses,
        nodes_count,
        ylabel="#Nodes",
        title="Diffusion Trend",
    ):
        """
        :param trends: Diffusion trends
        :param available_statuses: The available statuses
        :param nodes_count: nodes count
        :param ylabel: Label for Y axis
        :param title: Plot title
        """
        self.initialize_parent(trends, available_statuses, nodes_count)
        self.ylabel = ylabel
        self.title = title


class DiffusionPrevalencePlot(DiffusionPrevalence, NdlibDiffusionPlotMixin):
    def __init__(
        self,
        trends,
        available_statuses,
        nodes_count,
        ylabel="#Delta Nodes",
        title="Prevalence",
    ):
        """
        :param trends: Diffusion trends
        :param available_statuses: The available statuses
        :param nodes_count: nodes count
        :param ylabel: Label for Y axis
        :param title: Plot title
        """
        self.initialize_parent(trends, available_statuses, nodes_count)
        self.ylabel = ylabel
        self.title = title
        self.normalized = False


def plot_diffusion_trends(
    diffusion_model: DiffusionModel,
    raw_iterations: List[Dict],
    raw_trends: Optional[Dict] = None,
    filename: str = None,
):
    trends = raw_trends or diffusion_model.build_trends(raw_iterations)
    viz = DiffusionTrendPlot(
        trends=trends,
        available_statuses=diffusion_model.available_statuses,
        nodes_count=diffusion_model.graph.number_of_nodes(),
    )
    viz.plot(filename=filename)


def plot_diffusion_prevalence(
    diffusion_model: DiffusionModel,
    raw_iterations: List[Dict],
    raw_trends: Optional[Dict] = None,
    filename: str = None,
):
    trends = raw_trends or diffusion_model.build_trends(raw_iterations)
    viz = DiffusionPrevalencePlot(
        trends=trends,
        available_statuses=diffusion_model.available_statuses,
        nodes_count=diffusion_model.graph.number_of_nodes(),
    )
    viz.plot(filename=filename)


def plot_confusion_matrix(
    cm: ClassificationMetrics,
    title: str = "Confusion matrix",
    class_names=("Real", "Fake"),
    ax=None,
):
    matplotlib.use("Qt5Agg")
    disp = ConfusionMatrixDisplay(
        confusion_matrix=np.array(cm.confusion_matrix), display_labels=class_names
    )
    disp.plot(ax=ax)
    _configure_plot(title=title, xlabel="Detected sources", ylabel="Real sources")
