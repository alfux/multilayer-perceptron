import argparse as arg
import sys
import traceback
from typing import Self, Generator, Callable

import matplotlib.pyplot as plt
import matplotlib.colors as mclr
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.patches import PathPatch
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.widgets import Button
import pandas as pd
from pandas import DataFrame

from Statistics import Statistics
from PairPlot import PairPlot


class Visualizer:
    """Handle visualization of datas."""

    def __init__(self: Self, data: DataFrame, **param: dict) -> None:
        """Initializes the figue."""
        if "remap" in param:
            self._data = data
            self._remap = param["remap"]
        else:
            trait = param["trait"] if "trait" in param else '1'
            drop = param["drop"] if "drop" in param else ['0']
            self._data = self._pre_process(data, trait, drop)
        self._traits = list({x for x in self._data[0]})
        self._traits.sort()
        self._colors = list(mclr.TABLEAU_COLORS.values())[:len(self._traits)]
        self._no_describe = param.get("no_describe", False)
        title = param["title"] if "title" in param else "Visualizer"
        self._fig = plt.figure(title, figsize=(16, 9))
        self._fig.set_facecolor("0.85")
        (self._xaxis, self._yaxis) = (0, 0)
        self._button: list[Button] = list(self._generate_buttons())
        self._plot: Axes = self._fig.add_axes([0.04, 0.13, 0.9, 0.8])
        self._text: Text = self._fig.text(0.92, 0.1, self._describe())
        self._update_buttons()
        self._update_plot()

    def _pre_process(self: Self, data: DataFrame, trait: int,
                     drop: list) -> DataFrame:
        """Renames dataframe's headers."""
        traits = data[trait]
        head = {k: str(v) for (k, v) in zip(data.columns, data.columns)}
        data.rename(head, axis=1, inplace=True)
        data.drop([trait, *drop], axis=1, inplace=True)
        data = pd.concat([traits, data.select_dtypes("number")], axis=1)
        head = {k: v for (k, v) in zip(data.columns, range(len(data.columns)))}
        self._remap = {v: k for (v, k) in zip(head.values(), head.keys())}
        return data.rename(head, axis=1)

    def _describe(self: Self) -> str:
        """Data description text."""
        text = ''
        if self._no_describe:
            return
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

    def _generate_buttons(self: Self) -> Generator:
        """Generates buttons for the data visualization."""
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

    def _pair_plot(self: Self, event: MouseEvent) -> None:
        """Generates and displays a pair plot of the data."""
        if event.button == MouseButton.LEFT:
            PairPlot(self._data, remap=self._remap).show()

    def _box_plot(self: Self, event: MouseEvent) -> None:
        """Switch histogram and box plot."""
        if event.button == MouseButton.LEFT:
            self._box = not self._box
            if self._box:
                self._button[-1].label.set_text("Histogram")
            else:
                self._button[-1].label.set_text("Box Plot")
            self._update_buttons()
            self._update_plot()

    def _click_event(self: Self, column: int) -> Callable:
        """Generates on_click callbacks to update plotted datas."""

        def on_click(event: MouseEvent) -> None:
            """Updates collumns to show on screen."""
            if event.button == MouseButton.RIGHT:
                self._xaxis = column
            elif event.button == MouseButton.LEFT:
                self._yaxis = column
            self._update_buttons()
            self._update_plot()
            self._fig.canvas.draw()

        return on_click

    def _update_buttons(self: Self) -> None:
        """Updates buttons' colors."""
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

    def _display_stats(self: Self) -> None:
        """Displays statistics."""
        boxplot = self._plot.get_position()
        self._plot.set_position([boxplot.x0, boxplot.y0, 0.875, 0.8])
        self._text.set_text(self._describe())

    def _hide_stats(self: Self) -> None:
        """Hides statistics."""
        boxplot = self._plot.get_position()
        self._plot.set_position([boxplot.x0, boxplot.y0, 0.92, 0.8])
        self._text.set_text("")

    def _update_plot(self: Self) -> None:
        """Updates the plot on screen"""
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

    def _generate_histogram(self: Self) -> None:
        """Generates a histogram for the current selection."""
        for t in self._traits:
            select = [x for x in self._select(self._xaxis + 1, t)]
            self._plot.hist(select, bins=100, alpha=0.7)
        self._plot.legend(self._traits)
        self._plot.set_title(f"Distribution of {self._remap[self._xaxis + 1]}")

    def _generate_boxplot(self: Self) -> None:
        """Generates a box plot for the current selection."""

        def generate_selection() -> Generator:
            """Iterator of group of traits."""
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

    def _select(self: Self, column: int, trait: str) -> Generator:
        """Iterable selecting only values in column corresponding to trait."""
        for i in range(self._data.shape[0]):
            if self._data.at[i, 0] == trait:
                yield self._data.at[i, column]

    def show(self: Self) -> None:
        """Shows the visualizer."""
        plt.show()


def main() -> None:
    """Visualizes csv dataset."""
    try:
        parser = arg.ArgumentParser(description="Visualizes csv dataset")
        parser.add_argument("--debug", action="store_true", help="debug mode")
        parser.add_argument("--no-describe", action="store_true")
        parser.add_argument("data", help="csv dataset")
        parser.add_argument("drop", help="columns to drop, as an int index",
                            nargs="*", default=[])
        parser.add_argument("-t", "--trait", help="column of observed traits",
                            default=1)
        parser.add_argument("-n", "--no-header", help="first line is data",
                            action="store_true")
        parser = parser.parse_args()
        if parser.no_header:
            data = pd.read_csv(parser.data, header=None)
        else:
            data = pd.read_csv(parser.data)
        visualizer = Visualizer(data, trait=parser.trait,
                                drop=parser.drop,
                                title=parser.data,
                                no_describe=parser.no_describe)
        visualizer.show()
    except Exception as err:
        if parser.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
