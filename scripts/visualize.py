import argparse as arg
import sys
from typing import Self, Generator, Callable
import traceback

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent, MouseButton
from matplotlib.widgets import Button
from pandas import DataFrame, read_csv


class Visualizer:
    """Handle visualization of datas."""

    def __init__(self: Self, data: DataFrame) -> None:
        """Initializes the figue."""
        self._data = data
        self._fig = plt.figure(figsize=(16, 9))
        self._fig.set_facecolor("0.85")
        (self._xaxis, self._yaxis) = (2, 2)
        self._button: list[Button] = list(self._generate_buttons())
        self._plot: Axes = self._fig.add_axes([0.03, 0.13, 1 - 0.06, 1 - 0.2])
        self._update_buttons()
        self._update_plot()

    def _generate_buttons(self: Self) -> Generator:
        """Generates buttons for the data visualization."""
        n = self._data.shape[1] - 2
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
        else:
            self._button[self._xaxis].ax.set_facecolor((1, 0, 1, 0.7))
            self._button[self._xaxis].color = (1, 0, 1, 0.7)
            self._button[self._xaxis].hovercolor = (0.7, 0, 0.7, 0.7)
            self._button[self._yaxis].ax.set_facecolor((0, 1, 0, 0.7))
            self._button[self._yaxis].color = (0, 1, 0, 0.7)
            self._button[self._yaxis].hovercolor = (0, 0.7, 0, 0.7)

    def _update_plot(self: Self) -> None:
        """Updates the plot on screen"""
        self._plot.clear()
        if self._xaxis == self._yaxis:
            benign = list(self._select(self._xaxis + 2, "B"))
            malign = list(self._select(self._xaxis + 2, "M"))
            self._plot.hist(benign, bins=100, color=(0, 0.5, 1, 1))
            self._plot.hist(malign, bins=100, color=(1, 0, 0.1, 0.7))
            self._plot.legend(["Benign", "Malign"])
            self._plot.set_title(f"Distribution for character nº{self._xaxis}")
        else:
            benignx = list(self._select(self._xaxis + 2, "B"))
            benigny = list(self._select(self._yaxis + 2, "B"))
            malignx = list(self._select(self._xaxis + 2, "M"))
            maligny = list(self._select(self._yaxis + 2, "M"))
            self._plot.scatter(benignx, benigny, c=(0, 0.5, 1, 1))
            self._plot.scatter(malignx, maligny, c=(1, 0, 0.1, 0.7))
            self._plot.legend(["Benign", "Malign"])
            self._plot.set_title(f"{self._yaxis} against {self._xaxis}")
        self._fig.canvas.draw()

    def _select(self: Self, column: int, trait: str) -> Generator:
        """Iterable selecting only values in column corresponding to trait."""
        for i in range(self._data.shape[0]):
            if self._data.at[i, 1] == trait:
                yield self._data.at[i, column]

    def show(self: Self) -> None:
        """Shows the figure."""
        plt.show()


def main() -> None:
    """Visualizes csv dataset."""
    try:
        parser = arg.ArgumentParser(description="Visualizes csv dataset")
        parser.add_argument("data", help="csv dataset to visualize")
        data = read_csv(parser.parse_args().data, header=None)
        visualizer = Visualizer(data)
        visualizer.show()
    except Exception as err:
        print(traceback.format_exc())
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
