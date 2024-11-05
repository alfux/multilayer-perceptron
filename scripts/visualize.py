import argparse as arg
import sys
from typing import Self, Generator, Callable
import warnings

import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.widgets import Button
import pandas as pd
from pandas import DataFrame, read_csv

from describe import Statistics


class Visualizer:
    """Handle visualization of datas."""

    def __init__(self: Self, data: DataFrame, start: int, drop: list) -> None:
        """Initializes the figue."""
        self._data = self._rename(data, start, drop)
        self._traits = list({x for x in self._data[0]})
        self._stats = Statistics(self._data.drop(0, axis=1))
        self._fig = plt.figure(figsize=(16, 9))
        self._fig.set_facecolor("0.85")
        (self._xaxis, self._yaxis) = (0, 0)
        self._button: list[Button] = list(self._generate_buttons())
        self._plot: Axes = self._fig.add_axes([0.04, 0.13, 0.9, 0.8])
        self._text: Text = self._fig.text(0.92, 0.575, self._describe())
        self._update_buttons()
        self._update_plot()

    def _rename(self: Self, data: DataFrame, traits: int,
                drop: list) -> DataFrame:
        """Renames dataframe's headers."""
        head = {k: v for (k, v) in zip(data.columns, range(len(data.columns)))}
        data.rename(head, axis=1, inplace=True)
        traits = data[traits]
        data.drop(drop, axis=1, inplace=True)
        data = pd.concat([traits, data.select_dtypes("number")], axis=1)
        head = {k: v for (k, v) in zip(data.columns, range(len(data.columns)))}
        return data.rename(head, axis=1)

    def _describe(self: Self) -> str:
        """Data description text."""
        stats = self._stats.stats[self._xaxis]
        text = f"Count: {stats['N']:.3g}\n\n"
        text += f"Mean: {stats['Mean']:.3g}\n\n"
        text += f"Var: {stats['Var']:.3g}\n\n"
        text += f"Std: {stats['Std']:.3g}\n\n"
        text += f"Min: {stats['Min']:.3g}\n\n"
        text += f"25%: {stats['25%']:.3g}\n\n"
        text += f"50%: {stats['50%']:.3g}\n\n"
        text += f"75%: {stats['75%']:.3g}\n\n"
        text += f"Max: {stats['Max']:.3g}\n\n"
        return text

    def _generate_buttons(self: Self) -> Generator:
        """Generates buttons for the data visualization."""
        n = self._data.shape[1] - 1
        for i in range(n):
            axis = self._fig.add_axes(
                [(1 - 2e-2) * i / n + 1e-2, 0.04, (1 - 2e-2) / n, 0.03])
            button = Button(axis, i)
            button.on_clicked(self._click_event(i))
            yield button

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
        bbox = self._plot.get_position()
        self._plot.set_position([bbox.x0, bbox.y0, 0.875, 0.8])
        self._text.set_text(self._describe())

    def _hide_stats(self: Self) -> None:
        """Hides statistics."""
        bbox = self._plot.get_position()
        self._plot.set_position([bbox.x0, bbox.y0, 0.92, 0.8])
        self._text.set_text("")

    def _update_plot(self: Self) -> None:
        """Updates the plot on screen"""
        self._plot.clear()
        if self._xaxis == self._yaxis:
            for trait in self._traits:
                select = list(self._select(self._xaxis + 1, trait))
                self._plot.hist(select, bins=100, alpha=0.7)
            self._plot.legend(self._traits)
            self._plot.set_title(f"Distribution for character nº{self._xaxis}")
        else:
            for trait in self._traits:
                selectx = list(self._select(self._xaxis + 1, trait))
                selecty = list(self._select(self._yaxis + 1, trait))
                self._plot.scatter(selectx, selecty)
            self._plot.legend(self._traits)
            self._plot.set_title(f"{self._yaxis} against {self._xaxis}")
        self._fig.canvas.draw()

    def _select(self: Self, column: int, trait: str) -> Generator:
        """Iterable selecting only values in column corresponding to trait."""
        for i in range(self._data.shape[0]):
            if self._data.at[i, 0] == trait:
                yield self._data.at[i, column]

    def show(self: Self) -> None:
        """Shows the figure."""
        plt.show()


def main() -> None:
    """Visualizes csv dataset."""
    try:
        warnings.filterwarnings(action="ignore")
        parser = arg.ArgumentParser(description="Visualizes csv dataset")
        parser.add_argument("data", help="csv dataset")
        parser.add_argument("drop", help="columns to drop, as an int index",
                            nargs="*", type=int, default=0)
        parser.add_argument("-t", "--trait", help="column of observed traits",
                            type=int, default=1)
        parser.add_argument("-n", "--no-header", help="first line is data",
                            action="store_true")
        if parser.parse_args().no_header:
            data = read_csv(parser.parse_args().data, header=None)
        else:
            data = read_csv(parser.parse_args().data)
        visualizer = Visualizer(data, parser.parse_args().trait,
                                parser.parse_args().drop)
        visualizer.show()
    except Exception as err:
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
