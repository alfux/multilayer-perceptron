import argparse as arg
from argparse import Namespace
import logging
import sys


import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class Display:
    """Manage display of the perceptron cost functions."""

    _fig = None
    _axes = None
    _min = None
    _max = None
    _imax = None
    _packs = None
    _axes_max_zorder = None

    def __init__(
            self: "Display", color: str = "grey",
            margin: float = 0.1, name: str = "Display", batch: int = 1000
    ) -> None:
        """Initialize the display.

        Args:
            n (int): Number of curves. (default 1)
            margin (float): Top / bottem space margin. (default 0.1)
        """
        plt.ion()
        if Display._fig is None:
            self._init_display()
        self._name = name
        self._batch, self._curr_batch = batch, 0
        self._margin = margin
        self._i = [[[0] for _ in range(3)], [[0] for _ in range(3)]]
        self._v = [[[0] for _ in range(3)], [[0] for _ in range(3)]]
        self._color = color
        self._lines = self._create_lines()
        self._pack = {"legend": None, "single": set(), "move": set()}
        Display._packs.append(self._pack)
        if Display._axes[0].get_legend():
            Display._axes[0].get_legend().remove()
            Display._axes[1].get_legend().remove()
            self._draw()
        legend = Display._axes[0].legend()
        lines = legend.get_lines()
        for i, pack in enumerate(Display._packs):
            pack["legend"] = {lines[2 * i], lines[2 * i + 1]}
            lines[2 * i].set_picker(True)
            lines[2 * i + 1].set_picker(True)
            lines[2 * i].set_pickradius(5)
            lines[2 * i + 1].set_pickradius(5)
        Display._axes[1].legend()

    def loss(self: "Display", value: ndarray, i: int = 0) -> None:
        """Update the display with the new value.

        Args:
            value (ndarra): The new value.
            i (int): Curve index.
        """
        self._i[0][i].append(self._i[0][0][-1] + 1)
        self._v[0][i].append(value.astype(float))
        if value < Display._min[0]:
            Display._min[0] = value
        elif value > Display._max[0]:
            Display._max[0] = value
        ylim = (Display._min[0] - self._margin, Display._max[0] + self._margin)
        Display._imax[0] = np.max([Display._imax[0], self._i[0][i][-1]])
        self._axes[0].set_ylim(*ylim)
        self._axes[0].set_xlim(0, 1.1 * Display._imax[0])
        self._lines[0][i].set_data(self._i[0][i][1:], self._v[0][i][1:])
        if i == 0:
            self._lines[0][i].set_alpha(min(1, 5000 / Display._imax[0]))
        self._draw()

    def accuracy(self: "Display", value: ndarray, i: int = 1) -> None:
        """Update the display with the new value.

        Args:
            value (ndarra): The new value.
            i (int): Curve index.
        """
        self._i[1][i].append(self._i[0][0][-1])
        self._v[1][i].append(value.astype(float))
        if value < Display._min[1]:
            Display._min[1] = value
        elif value > Display._max[1]:
            Display._max[1] = value
        ylim = (Display._min[1] - self._margin, Display._max[1] + self._margin)
        Display._imax[1] = np.max([Display._imax[1], self._i[1][i][-1]])
        self._axes[1].set_ylim(*ylim)
        self._axes[1].set_xlim(0, 1.1 * Display._imax[1])
        self._lines[1][i].set_data(self._i[1][i][1:], self._v[1][i][1:])
        self._draw()

    def metrics(self: "Display", **metrics: dict) -> None:
        """Separator line for epochs.

        Args:
            x (float): The abscissa of the vertical line.
        """
        abs = self._i[0][1][-1]
        Display._axes[0].axvline(abs, color=self._color, zorder=0, alpha=0.25)
        text = Display._axes[0].annotate(
            Display.format_dict(metrics),
            xy=(abs, self._max[0] - self._margin * 2),
            xycoords="data",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor=self._color,
                facecolor="white",
            ),
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=Display._axes_max_zorder + 1,
            picker=True
        )
        self._pack["move"].add(text)
        self._pack["single"].add(text)
        self._draw()

    def _init_display(self: "Display") -> None:
        """Initialize the display class"""
        Display._fig = [
            plt.figure(figsize=(16, 9)), plt.figure(figsize=(8, 4.5))
        ]
        Display._axes = [
            Display._fig[0].add_axes((0.1, 0.1, 0.8, 0.8)),
            Display._fig[1].add_axes((0.1, 0.1, 0.8, 0.8))
        ]
        axis1, axis2 = Display._axes
        Display._axes_max_zorder = 0
        for s1, s2 in zip(axis1.spines.values(), axis2.spines.values()):
            if Display._axes_max_zorder < s1.zorder:
                Display._axes_max_zorder = s1.zorder
            if Display._axes_max_zorder < s2.zorder:
                Display._axes_max_zorder = s2.zorder
        Display._min, Display._max, Display._imax = [0, 0], [0, 0], [100, 100]
        Display._fig[0].canvas.manager.set_window_title("Loss")
        Display._fig[1].canvas.manager.set_window_title("Accuracy")
        Display._axes[0].set_title("Loss")
        Display._axes[1].set_title("Accuracy")
        Display._packs = []
        Display._fig[0].canvas.mpl_connect("pick_event", Display.picker_hook)

    def _create_lines(self: "Display") -> list:
        """Creates lines structure.

        Returns:
            list: Lines structure.
        """
        deloss = self._axes[0].plot(
            self._i[0][0], self._v[0][0], color=self._color, linewidth=2,
            zorder=0, label=self._name)
        veloss = self._axes[0].plot(
            self._i[0][0], self._v[0][0], color=self._color, linewidth=2,
            zorder=0, label=self._name, linestyle="--")
        iloss = self._axes[0].plot(
            self._i[0][1], self._v[0][1], color=self._color, linewidth=0.1,
            zorder=0)
        dacc = self._axes[1].plot(
            self._i[1][0], self._v[1][0], color=self._color, linewidth=0.5,
            zorder=0, label=(self._name + " data accuracy"))
        vacc = self._axes[1].plot(
            self._i[1][1], self._v[1][1], color=self._color, linewidth=2,
            zorder=0, label=(self._name + " validation accuracy"))
        return [[iloss[0], deloss[0], veloss[0]], [dacc[0], vacc[0]]]

    def _draw(self: "Display") -> None:
        """Draw to the screen."""
        if self._curr_batch < self._batch:
            self._curr_batch += 1
        else:
            plt.draw()
            plt.pause(1e-15)
            self._curr_batch = 0

    @staticmethod
    def format_dict(jso: dict) -> str:
        """Format a dict in a line by line string.

        Args:
            jso (dict): The dictionary to format.
        Returns:
            str: The formated string.
        """
        string = ''
        sep, i = [' ', '\n'], 0
        for key, value in jso.items():
            if isinstance(value, float):
                value = "{:.2f}".format(value)
            else:
                value = str(value)
            string += str(key) + ': ' + str(value) + sep[i]
            i = (i + 1) % 2
        return string

    @staticmethod
    def picker_hook(evt) -> None:
        """Hook for hover to foreground readings.

        Args:
            evt (): The event triggering the hook.
        """
        artist = evt.artist
        for pack in Display._packs:
            if artist in pack["legend"]:
                Display.move_artists(pack["move"], 2)
            elif artist in pack["single"]:
                Display.move_artists(pack["move"], 2)
                artist.set_zorder(Display._axes_max_zorder + 3)
            else:
                Display.move_artists(pack["move"], 1)
        plt.draw()

    @staticmethod
    def move_artists(pack: set, z: int) -> None:
        """Move artists according to pack and zorder.

        Args:
            pack (set): The pack of artists to move.
            z (int): zorder over the max to put tem in.
        """
        for art in pack:
            art.set_zorder(Display._axes_max_zorder + z)
            art.set_visible(z == 2)

    @staticmethod
    def pause() -> None:
        """Pauses the program until display is closed."""
        if Display._fig is not None:
            plt.ioff()
            plt.show()
            plt.ion()


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_irue", help="Traceback mode.")
    return av.parse_args()


def main() -> int:
    """Test main.

    Returns:
        int: return status 0 (success) 1 (error).
    """
    try:
        av = get_args(main.__doc__)
        fmt = "%(asctime)s | %(levelname)s - %(message)s"
        if av.debug:
            logging.basicConfig(level=logging.DEBUG, format=fmt)
        else:
            logging.basicConfig(level=logging.INFO, format=fmt)
        logging.info("This module is not testable alone at the moment.")
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
