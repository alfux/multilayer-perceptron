import argparse as arg
import sys
import traceback
from typing import Generator, Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.patches import PathPatch
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.widgets import Button
import pandas as pd
from pandas import DataFrame

from .statistics import Statistics
from .pair_plot import PairPlot


class Visualizer:
    """Interactive data visualizer for numeric features."""

    def __init__(self: "Visualizer", data: DataFrame, **param: dict) -> None:
        """Initialize the figure and UI elements.

        Args:
            data (DataFrame): Input data.
            **param: Either ``remap`` mapping, or ``trait`` and ``drop``
                for preprocessing, plus optional ``title``.
        """
        if "remap" in param:
            self._data = data
            self._remap = param["remap"]
        else:
            trait = param.get("trait", None)
            drop = param.get("drop", ['0'])
            self._data = self._pre_process(data, trait, drop)
        self._traits = list({x for x in self._data[0]})
        self._traits.sort()
        self._colors = list(mclr.TABLEAU_COLORS.values())[:len(self._traits)]
        title = param["title"] if "title" in param else "Visualizer"
        self._fig = plt.figure(title, figsize=(16, 9))
        self._fig.set_facecolor("0.85")
        (self._xaxis, self._yaxis) = (0, 0)
        self._button: list[Button] = list(self._generate_buttons())
        self._plot: Axes = self._fig.add_axes([0.04, 0.13, 0.9, 0.8])
        self._text: Text = self._fig.text(0.92, 0.1, self._describe())
        self._update_buttons()
        self._update_plot()

    def _pre_process(self: "Visualizer", data: DataFrame, trait: int,
                     drop: list) -> DataFrame:
        """Rename DataFrame headers and select numeric columns."""
        head = {k: str(v) for (k, v) in zip(data.columns, data.columns)}
        data.rename(head, axis=1, inplace=True)
        if trait is not None:
            traits = data[trait]
            data.drop([trait, *drop], axis=1, inplace=True)
        else:
            traits = DataFrame({"None": ['None'] * data.shape[0]})
            data.drop([*drop], axis=1, inplace=True)
        data = pd.concat([traits, data.select_dtypes("number")], axis=1)
        head = {k: v for (k, v) in zip(data.columns, range(len(data.columns)))}
        self._remap = {v: k for (v, k) in zip(head.values(), head.keys())}
        return data.rename(head, axis=1)

    def _describe(self: "Visualizer") -> str:
        """Generate text description (statistics) for the selected axis."""
        text = ''
        for trait in self._traits:
            selected = self._select(self._xaxis + 1, trait)
            stats = Statistics(DataFrame(selected)).stats
            text += f"{trait}\n\n"
            text += f"Count: {stats.loc['N', 0]:.3g}\n\n"
            text += f"Mean: {stats.loc['Mean', 0]:.3g}\n\n"
            text += f"Var: {stats.loc['Var', 0]:.3g}\n\n"
            text += f"Std: {stats.loc['Std', 0]:.3g}\n\n"
            text += f"Min: {stats.loc['Min', 0]:.3g}\n\n"
            text += f"25%: {stats.loc['25%', 0]:.3g}\n\n"
            text += f"50%: {stats.loc['50%', 0]:.3g}\n\n"
            text += f"75%: {stats.loc['75%', 0]:.3g}\n\n"
            text += f"Max: {stats.loc['Max', 0]:.3g}\n\n\n"
        return text

    def _generate_buttons(self: "Visualizer") -> Generator:
        """Generate control buttons for the visualization."""
        n = self._data.shape[1] - 1
        for i in range(n):
            axis = self._fig.add_axes(
                [(1 - 2e-2) * i / n + 1e-2, 0.02, (1 - 2e-2) / n, 0.03])
            button = Button(axis, self._remap[i + 1])
            button.on_clicked(self._click_event(i))
            yield button
        axis = self._fig.add_axes([0.14, 0.07, 0.1, 0.03])
        button = Button(axis, "Pair Plot")
        button.on_clicked(lambda event: self._pair_plot(event))
        yield button
        self._box = False
        axis = self._fig.add_axes([0.04, 0.07, 0.1, 0.03])
        button = Button(axis, "Box Plot")
        button.on_clicked(lambda event: self._box_plot(event))
        yield button

    def _pair_plot(self: "Visualizer", event: MouseEvent) -> None:
        """Generate and display a pair plot of the data."""
        if event.button == MouseButton.LEFT:
            PairPlot(self._data, remap=self._remap).show()

    def _box_plot(self: "Visualizer", event: MouseEvent) -> None:
        """Toggle between histogram and box plot modes."""
        if event.button == MouseButton.LEFT:
            self._box = not self._box
            if self._box:
                self._button[-1].label.set_text("Histogram")
            else:
                self._button[-1].label.set_text("Box Plot")
            self._update_buttons()
            self._update_plot()

    def _click_event(self: "Visualizer", column: int) -> Callable:
        """Generate callbacks to update plotted data on click."""

        def on_click(event: MouseEvent) -> None:
            """Update selected columns for display."""
            if event.button == MouseButton.RIGHT:
                self._xaxis = column
            elif event.button == MouseButton.LEFT:
                self._yaxis = column
            self._update_buttons()
            self._update_plot()
            self._fig.canvas.draw()

        return on_click

    def _update_buttons(self: "Visualizer") -> None:
        """Update buttons' colors to reflect selection."""
        for button in self._button:
            button.ax.set_facecolor("1")
            button.color = "1"
            button.hovercolor = "0.85"
        if self._xaxis == self._yaxis:
            self._button[self._xaxis].ax.set_facecolor((1, 1, 0, 0.7))
            self._button[self._xaxis].color = (1, 1, 0, 0.7)
            self._button[self._xaxis].hovercolor = (0.7, 0.7, 0, 0.7)
            self._display_stats()
        else:
            self._button[self._xaxis].ax.set_facecolor((1, 0, 1, 0.7))
            self._button[self._xaxis].color = (1, 0, 1, 0.7)
            self._button[self._xaxis].hovercolor = (0.7, 0, 0.7, 0.7)
            self._button[self._yaxis].ax.set_facecolor((0, 1, 0, 0.7))
            self._button[self._yaxis].color = (0, 1, 0, 0.7)
            self._button[self._yaxis].hovercolor = (0, 0.7, 0, 0.7)
            self._hide_stats()

    def _display_stats(self: "Visualizer") -> None:
        """Display statistics pane."""
        boxplot = self._plot.get_position()
        self._plot.set_position([boxplot.x0, boxplot.y0, 0.875, 0.8])
        self._text.set_text(self._describe())

    def _hide_stats(self: "Visualizer") -> None:
        """Hide statistics pane."""
        boxplot = self._plot.get_position()
        self._plot.set_position([boxplot.x0, boxplot.y0, 0.92, 0.8])
        self._text.set_text("")

    def _update_plot(self: "Visualizer") -> None:
        """Update the plot on screen."""
        self._plot.clear()
        if self._xaxis == self._yaxis:
            if self._box:
                self._generate_boxplot()
                pass
            else:
                self._generate_histogram()
        else:
            for trait in self._traits:
                selectx = list(self._select(self._xaxis + 1, trait))
                selecty = list(self._select(self._yaxis + 1, trait))
                self._plot.scatter(selectx, selecty)
            self._plot.legend(self._traits)
            self._plot.set_title((f"{self._remap[self._yaxis + 1]} against"
                                  + f" {self._remap[self._xaxis + 1]}"))
        self._fig.canvas.draw()

    def _generate_histogram(self: "Visualizer") -> None:
        """Generate a histogram for the current selection."""
        for t in self._traits:
            select = [x for x in self._select(self._xaxis + 1, t)]
            self._plot.hist(select, bins=100, alpha=0.7)
        self._plot.legend(self._traits)
        self._plot.set_title(f"Distribution of {self._remap[self._xaxis + 1]}")

    def _generate_boxplot(self: "Visualizer") -> None:
        """Generate a box plot for the current selection."""

        def generate_selection() -> Generator:
            """Iterate groups of traits for box plotting."""
            for t in self._traits:
                yield [x for x in self._select(self._xaxis + 1, t) if x == x]

        param = {"vert": False, "widths": 0.5, "meanline": True,
                 "showmeans": True, "patch_artist": True,
                 "tick_labels": self._traits[::-1]}
        boxplot = self._plot.boxplot(list(generate_selection())[::-1], **param)
        box: PathPatch
        for (box, color) in zip(boxplot["boxes"], self._colors[::-1]):
            box.set_facecolor(color)
            box.set_alpha(0.4)
        self._plot.set_title(f"Distribution of {self._remap[self._xaxis + 1]}")

    def _select(self: "Visualizer", column: int, trait: str) -> Generator:
        """Iterate values in ``column`` corresponding to the given trait."""
        for i in range(self._data.shape[0]):
            if self._data.at[i, 0] == trait:
                yield self._data.at[i, column]

    def show(self: "Visualizer") -> None:
        """Show the visualizer window."""
        plt.show()


def main() -> None:
    """Visualize a CSV dataset with interactive controls."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("--sample", type=float, default=1, help="sample size")
        av.add_argument("data", help="csv dataset")
        message = "columns to drop, as an int index"
        av.add_argument("drop", help=message, nargs="*", default=[])
        message = "column of observed traits"
        av.add_argument("-t", "--trait", help=message, default=None)
        message = "first line is data"
        av.add_argument("-n", "--no-header", help=message, action="store_true")
        av = av.parse_args()
        if av.no_header:
            data = pd.read_csv(av.data, header=None).sample(frac=av.sample)
        else:
            data = pd.read_csv(av.data).sample(frac=av.sample)
        data = data.reset_index(drop=True)
        Visualizer(data, trait=av.trait, drop=av.drop, title=av.data).show()
    except Exception as err:
        if av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
