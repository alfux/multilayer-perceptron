import argparse as arg
import os
import sys
import traceback

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from MLP import MLP


GRID = 100


def plot_mlp(path: str, ax: Axes) -> None:
    """Plots the mlp's output."""
    mlp: MLP = MLP.load(path)
    (x, y) = (np.linspace(0, 86400, GRID), np.linspace(0, 7500, GRID))
    (x, y) = np.meshgrid(x, y)
    z = np.array([[None] * GRID] * GRID)
    for i in range(GRID):
        input_layer = np.concat([np.atleast_2d(x[i]), np.atleast_2d(y[i])])
        z[i] = mlp.eval(input_layer.T).T
    ax.plot_surface(x, y, z, cmap="viridis")


def plot_training_set(path: str, ax: Axes) -> None:
    """Plots the training set values."""
    for base in os.listdir(path):
        base = path + "/" + base
        base = pd.read_csv(base).loc[:, ["IEA", "Temps (sec)", "FaitJour"]]
        (x, y, z) = (base["Temps (sec)"], base["FaitJour"], base["IEA"])
        (x, y, z) = (x.to_numpy(), y.to_numpy(), z.to_numpy())
        grid = int(np.ceil(len(x) / GRID))
        (x, y, z) = (x[::grid], y[::grid], z[::grid])
        ax.plot(x, y, z, linewidth=1)


def main() -> int:
    """Plot the model's surface against the training datas."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("mlp", help="path to mlp file")
        av.add_argument("base", help="path to the base")
        av = av.parse_args()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        plot_mlp(av.mlp, ax)
        plot_training_set(av.base, ax)
        plt.title("Learned curve against real curves")
        plt.show()
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
