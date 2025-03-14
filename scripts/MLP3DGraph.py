import argparse as arg
import sys
import traceback
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from MLP import MLP


class MLP3DGraph:
    """Used to plot a 3D Graph projection of the MLP against training set."""

    def __init__(
            self: Self, mlp: str | MLP, training: str | DataFrame,
            frac: float = 1, title: str = "Graph", grid: int = 100,
            figsize: tuple = (15, 9), alpha: float = 0.75
    ) -> None:
        """Initializes the 3D Graph projection.

        <b>Args:</b>
            <b>mlp</b> is the path of the MLP or the MLP
            <b>training</b> is the path of the training set or the training set
            <b>frac</b> is the ratio of the sample
            <b>title</b> is the window title
            <b>grid</b> is the mesh smoothness
            <b>figsize</b> is the window size
        <b>Returns:</b>
            None
        """
        self._fig: Figure = plt.figure(figsize=figsize)
        self._ax: Axes = self._fig.add_subplot(111, projection="3d")
        if isinstance(mlp, str):
            self._mlp: MLP = MLP.load(mlp)
        else:
            self._mlp = mlp
        if isinstance(training, str):
            self._set: DataFrame = pd.read_csv(training).sample(frac=frac)
        else:
            self._set = training
        self._grid: int = grid
        self._alpha: float = alpha
        self._x: ndarray = self._set["Temps (sec)"].to_numpy()
        self._y: ndarray = self._set["FaitJour"].to_numpy()
        self._z: ndarray = self._set["IEA"].to_numpy()
        self._fig.suptitle(title)
        self._ax.set_xlabel("Temps (sec)")
        self._ax.set_ylabel("FaitJour")
        self._ax.set_zlabel("Index Energie Active")

    def plot_mlp(self: Self) -> Self:
        """Plots the mlp's output.

        <b>Returns:</b>
            The current instance
        """
        x = np.linspace(np.min(self._x), np.max(self._x), self._grid)
        y = np.linspace(np.min(self._y), np.max(self._y), self._grid)
        (x, y) = np.meshgrid(x, y)
        z = np.array([[None] * self._grid] * self._grid)
        for i in range(self._grid):
            input_layer = np.concat([np.atleast_2d(x[i]), np.atleast_2d(y[i])])
            z[i] = self._mlp.eval(input_layer.T).T
        self._ax.plot_surface(x, y, z, cmap="viridis", alpha=self._alpha)
        return self

    def plot_training_set(self: Self) -> Self:
        """Plots the training set values.

        <b>Returns:</b>
            The current instance
        """
        grid = int(np.ceil(len(self._x) / self._grid))
        (x, y, z) = (self._x[::grid], self._y[::grid], self._z[::grid])
        self._ax.scatter(x, y, z, c=z, cmap="viridis", marker="o", alpha=1)
        return self

    def verify(self: Self) -> Self:
        """Verifies the score"""
        input_layer = self._set.loc[:, ["Temps (sec)", "FaitJour"]].to_numpy()
        truth = self._set.loc[:, ["IEA"]].to_numpy()
        output = self._mlp.eval(input_layer)
        print("\n\tLoss = ", self._mlp.cost.eval(truth, output))


def main() -> int:
    """Plot the model's surface against the training datas."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        message = "ratio of the sample compared to the full set of data"
        av.add_argument("--sample", default=1, type=float, help=message)
        message = "alpha value of the mlp surface (transparency)"
        av.add_argument("--alpha", default=1.0, type=float, help=message)
        message = "axes subdivisions to determine surface mesh subdivisions"
        av.add_argument("--grid", default=100, type=int, help=message)
        av.add_argument("training", help="path to the training file")
        av.add_argument("mlp", help="path to mlp file")
        av = av.parse_args()
        graph = MLP3DGraph(
            av.mlp, av.training, av.sample, title="Consommation étalon",
            alpha=av.alpha, grid=av.grid
        )
        graph.plot_training_set().plot_mlp().verify()
        plt.show()
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
