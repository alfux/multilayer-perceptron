"""Public API for the ``alfux-mlp`` package.

Exports core classes such as ``Neuron``, ``Layer``, ``MLP``, and utilities for
training, visualization, and statistics.
"""
from .neuron import Neuron
from .layer import Layer
from .mlp import MLP
from .teacher import Teacher
from .mlp_graph3d import MLP3DGraph
from .pair_plot import PairPlot
from .processor import Processor
from .statistics import Statistics
from .visualizer import Visualizer

__all__ = [
    "Neuron", "Layer", "MLP", "Teacher", "MLP3DGraph", "PairPlot", "Processor",
    "Statistics", "Visualizer", "Regression"
]
