import argparse as arg
import sys
from typing import Generator

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from pandas import DataFrame


class PairPlot:
    """Generate a pair plot from a DataFrame."""

    def __init__(self: "PairPlot", data: DataFrame, **kwargs: dict):
        """Process the DataFrame and create the pair plot.

        Args:
            data (DataFrame): Input data.
            **kwargs: Either ``remap`` mapping, or ``trait`` and ``drop``
                configuration to preprocess columns.
        """
        if "remap" in kwargs:
            self._data = data
            self._remap = kwargs["remap"]
        else:
            trait = kwargs["trait"] if "trait" in kwargs else '1'
            drop = kwargs["drop"] if "drop" in kwargs else ['0']
            self._data = self._pre_process(data, trait, drop)
        self._traits = list({x for x in self._data[0]})
        self._traits.sort()
        self._fig = plt.figure("Pair Plot", figsize=(16, 9))
        self._fig.set_facecolor("0.85")
        self._fig.subplots_adjust(0.02, 0.01, 0.98, 0.93)
        self._generate_pair_plot()

    def _pre_process(self: "PairPlot", data: DataFrame, trait: int,
                     drop: list) -> DataFrame:
        """Rename DataFrame headers and select numeric columns."""
        traits = data[trait]
        head = {k: str(v) for (k, v) in zip(data.columns, data.columns)}
        data.rename(head, axis=1, inplace=True)
        data.drop(drop, axis=1, inplace=True)
        data = pd.concat([traits, data.select_dtypes("number")], axis=1)
        head = {k: v for (k, v) in zip(data.columns, range(len(data.columns)))}
        self._remap = {v: k for (v, k) in zip(head.values(), head.keys())}
        return data.rename(head, axis=1)

    def _generate_pair_plot(self: "PairPlot") -> None:
        """Generate all subplots for the pair plot."""
        labels = self._data.columns[1:]
        mosaic = [[f"{i}/{j}" for j in labels] for i in labels]
        plots: dict[str, Axes] = self._fig.subplot_mosaic(mosaic)
        for i in labels:
            for j in labels:
                if i != j:
                    self._scatter_plot(plots[f"{i}/{j}"], i, j)
                else:
                    self._histogram(plots[f"{i}/{j}"], i)
        self._fig.legend(self._traits, fancybox=True, shadow=True, ncol=4,
                         loc="upper center")

    def _scatter_plot(self: "PairPlot", plot: Axes, i: int, j: int) -> None:
        """Generate a scatter plot of feature ``i`` against ``j``."""
        for trait in self._traits:
            selectx = list(self._select(j, trait))
            selecty = list(self._select(i, trait))
            plot.scatter(selectx, selecty,
                         s=(100 / (self._data.shape[1] - 1) ** 2))
        plot.set_xticks([])
        plot.set_yticks([])
        if i == self._data.columns[1]:
            plot.set_xlabel(self._remap[j], loc="center")
            plot.xaxis.set_label_position("top")
        if j == self._data.columns[1]:
            plot.set_ylabel(self._remap[i], loc="center")
            plot.yaxis.set_label_position("left")

    def _histogram(self: "PairPlot", plot: Axes, i: int) -> None:
        """Generate a histogram of feature ``i``."""
        for trait in self._traits:
            selected = list(self._select(i, trait))
            plot.hist(selected, bins=100)
        plot.set_xticks([])
        plot.set_yticks([])
        if i == self._data.columns[1]:
            plot.set_xlabel(self._remap[i], loc="center")
            plot.xaxis.set_label_position("top")
            plot.set_ylabel(self._remap[i], loc="center")
            plot.yaxis.set_label_position("left")

    def _select(self: "PairPlot", column: int, trait: str) -> Generator:
        """Iterate values in ``column`` corresponding to the given trait."""
        for i in range(self._data.shape[0]):
            if self._data.at[i, 0] == trait:
                yield self._data.at[i, column]

    def show(self: "PairPlot") -> None:
        """Show the pair plot."""
        plt.show()


def main() -> None:
    """Display a pair plot from the given CSV file."""
    try:
        parser = arg.ArgumentParser(description="Visualizes csv dataset")
        parser.add_argument("data", help="csv dataset")
        parser.add_argument("drop", help="columns to drop, as an int index",
                            nargs="*", default='0')
        parser.add_argument("-t", "--trait", help="column of observed traits",
                            default=1)
        parser.add_argument("-n", "--no-header", help="first line is data",
                            action="store_true")
        if parser.parse_args().no_header:
            data = pd.read_csv(parser.parse_args().data, header=None)
        else:
            data = pd.read_csv(parser.parse_args().data)
        pair_plot = PairPlot(data, trait=parser.parse_args().trait,
                             drop=parser.parse_args().drop)
        pair_plot.show()

    except Exception as err:
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
