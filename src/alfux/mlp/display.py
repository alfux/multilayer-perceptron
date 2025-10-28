import argparse as arg
from argparse import Namespace
import logging
import sys
import time


import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray


class Display:
    """Manage display of the perceptron cost functions."""

    _fig = None
    _axes = None
    _min = None
    _max = None
    _tmax = None
    _packs = None
    _axes_max_zorder = None

    def __init__(
            self: "Display", n: int = 1, color: str = "grey", margin: float = 0.1,
            title: str = "Display"
    ) -> None:
        """Initialize the display.

        Args:
            n (int): Number of curves. (default 1)
            margin (float): Top / bottem space margin. (default 0.1)
        """
        plt.ion()
        if Display._fig is None:
            self._init_display(title)
        self._start, self._margin = time.time(), margin
        self._t = [[[] for _ in range(n)], [[] for _ in range(n)]]
        self._v = [[[] for _ in range(n)], [[] for _ in range(n)]]
        self._color = []
        self._lines = self._create_lines()
        self._pack = {"click": [_ for _ in self._lines[0]], "move": []}
        Display._packs.append(self._pack)
        if Display._axes[0].get_legend():
            Display._axes[0].get_legend().remove()
            Display._axes[1].get_legend().remove()
            self._draw()
        Display._axes[0].legend()
        Display._axes[1].legend()

    def loss(self: "Display", value: ndarray, i: int = 0) -> None:
        """Update the display with the new value.

        Args:
            value (ndarra): The new value.
            i (int): Curve index.
        """
        self._t[0][i].append(time.time() - self._start)
        self._v[0][i].append(value.astype(float))
        if value < Display._min[0]:
            Display._min[0] = value
        elif value > Display._max[0]:
            Display._max[0] = value
        ylim = (Display._min[0] - self._margin, Display._max[0] + self._margin)
        Display._tmax[0] = np.max([Display._tmax[0], self._t[0][i][-1]])
        self._axes[0].set_ylim(*ylim)
        self._axes[0].set_xlim(0, 1.1 * Display._tmax[0])
        self._lines[0][i].set_data(self._t[0][i], self._v[0][i])
        self._draw()

    def accuracy(self: "Display", value: ndarray, i: int = 1) -> None:
        """Update the display with the new value.

        Args:
            value (ndarra): The new value.
            i (int): Curve index.
        """
        self._t[1][i].append(time.time() - self._start)
        self._v[1][i].append(value.astype(float))
        if value < Display._min[1]:
            Display._min[1] = value
        elif value > Display._max[1]:
            Display._max[1] = value
        ylim = (Display._min[1] - self._margin, Display._max[1] + self._margin)
        Display._tmax[1] = np.max([Display._tmax[1], self._t[1][i][-1]])
        self._axes[1].set_ylim(*ylim)
        self._axes[1].set_xlim(0, 1.1 * Display._tmax[1])
        self._lines[1][i].set_data(self._t[1][i], self._v[1][i])
        self._draw()

    def metrics(self: "Display", **metrics: dict) -> None:
        """Separator line for epochs.

        Args:
            x (float): The abscissa of the vertical line.
        """
        abs = time.time() - self._start
        color = self._param[-1].get("color", "grey")
        vline = Display._axes[0].axvline(
            abs, color=color, picker=True, zorder=0, alpha=0.25
        )
        text = Display._axes[0].annotate(
            Display.format_dict(metrics),
            xy=(abs, self._max[0]),
            xycoords="data",
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor=color,
                facecolor="white",
            ),
            ha="left",
            va="bottom",
            zorder=Display._axes_max_zorder + 1,
            picker=True,
        )
        self._pack["click"] += [vline, text]
        self._pack["move"] += [text]
        self._draw()

    def _init_display(self: "Display", title: str) -> None:
        """Initialize the display class.

        Args:
            title (str): Title of the display.
        """
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
        Display._min, Display._max, Display._tmax = [0, 0], [0, 0], [30, 30]
        Display._fig[0].canvas.manager.set_window_title(title)
        Display._fig[1].canvas.manager.set_window_title(title)
        Display._axes[0].set_title(title)
        Display._axes[1].set_title(title)
        Display._packs = []
        Display._fig[0].canvas.mpl_connect("pick_event", Display.picker_hook)

    def _create_lines(self: "Display") -> list:
        """Creates lines structure.

        Returns:
            list: Lines structure.
        """
        return [
            [
                self._axes[0].plot(t, v, color=self._,
                                   picker=True, zorder=0)[0]
                for t, v, p in zip(self._t[0], self._v[0], self._param)
            ],
            [
                self._axes[1].plot(t, v, **p, picker=True, zorder=0)[0]
                for t, v, p in zip(self._t[1], self._v[1], self._param)
            ]
        ]

    def _draw(self: "Display") -> None:
        """Draw to the screen."""
        plt.draw()
        plt.pause(1e-15)

    @staticmethod
    def format_dict(jso: dict) -> str:
        """Format a dict in a line by line string.

        Args:
            jso (dict): The dictionary to format.
        Returns:
            str: The formated string.
        """
        string = ''
        for key, value in jso.items():
            if isinstance(value, float):
                value = "{:.2f}".format(value)
            else:
                value = str(value)
            string += str(key) + ': ' + str(value) + '\n'
        return string

    @staticmethod
    def picker_hook(evt) -> None:
        """Hook for hover to foreground readings.

        Args:
            evt (): The event triggering the hook.
        """
        artist = evt.artist
        for pack in Display._packs:
            if artist in pack["click"]:
                for art in pack["move"]:
                    art.set_zorder(Display._axes_max_zorder + 2)
            else:
                for art in pack["move"]:
                    art.set_zorder(Display._axes_max_zorder + 1)
        plt.draw()

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
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
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
