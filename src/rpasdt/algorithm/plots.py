"""Plots utilities."""
from typing import Dict, List, Optional

import matplotlib
from ndlib.models.DiffusionModel import DiffusionModel
from ndlib.viz.mpl.DiffusionPrevalence import DiffusionPrevalence
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend

matplotlib.use("Qt5Agg")


def plot_diffusion_trends(
    diffusion_model: DiffusionModel,
    raw_iterations: List[Dict],
    raw_trends: Optional[Dict] = None,
    filename: str = None,
):
    trends = raw_trends or diffusion_model.build_trends(raw_iterations)
    viz = DiffusionTrend(diffusion_model, trends)
    viz.plot(filename=filename)


def plot_diffusion_prevalence(
    diffusion_model: DiffusionModel,
    raw_iterations: List[Dict],
    raw_trends: Optional[Dict] = None,
    filename: str = None,
):
    trends = raw_trends or diffusion_model.build_trends(raw_iterations)
    viz = DiffusionPrevalence(diffusion_model, trends)
    viz.plot(filename=filename)
