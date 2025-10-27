import argparse as arg
from argparse import Namespace
import logging
import sys
import time


import matplotlib.pyplot as plt
import matplotlib.patches as pch
# from matplotlib import Event
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

    def __init__(
            self: "Display", n: int = 1, param: list = [], margin: float = 0.1,
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
        self._times = [[] for _ in range(n)]
        self._values = [[] for _ in range(n)]
        if len(param) < n:
            param += [{}] * (n - len(param))
        self._param = param
        self._lines = [
            self._axes.plot(t, v, **p, picker=True, zorder=0)[0]
            for t, v, p in zip(self._times, self._values, self._param)
        ]
        self._pack = {"click": [_ for _ in self._lines], "move": []}
        Display._packs.append(self._pack)
        if Display._axes.get_legend():
            Display._axes.get_legend().remove()
            plt.draw()
        Display._axes.legend()

    def __call__(self: "Display", value: ndarray, i: int = 0) -> None:
        """Update the display with the new value.

        Args:
            value (ndarra): The new value.
            i (int): Curve index.
        """
        self._times[i].append(time.time() - self._start)
        self._values[i].append(value.astype(float))
        if value < Display._min:
            Display._min = value
        elif value > Display._max:
            Display._max = value
        ylim = (Display._min - self._margin, Display._max + self._margin)
        Display._tmax = np.max([Display._tmax, self._times[i][-1]])
        self._axes.set_ylim(*ylim)
        self._axes.set_xlim(0, Display._tmax)
        self._lines[i].set_data(self._times[i], self._values[i])
        plt.draw()
        plt.pause(1e-15)

    def metrics(self: "Display", **metrics: dict) -> None:
        """Separator line for epochs.

        Args:
            x (float): The abscissa of the vertical line.
        """
        abs = time.time() - self._start
        color = self._param[-1].get("color", "grey")
        anchor = (abs, 0.2)
        rect = pch.Rectangle(
            anchor, 2.5, 0.1, edgecolor=color, facecolor="white", picker=True,
            zorder=1
        )
        vline = Display._axes.axvline(
            abs, color=color, picker=True, zorder=0, alpha=0.25
        )
        patch = Display._axes.add_patch(rect)
        text = Display._axes.text(
            *anchor, Display.format_dict(metrics), picker=True, zorder=1
        )
        self._pack["click"] += [vline, patch, text]
        self._pack["move"] += [patch, text]
        plt.draw()
        plt.pause(1e-15)

    def _init_display(self: "Display", title: str) -> None:
        """Initialize the display class.

        Args:
            title (str): Title of the display.
        """
        Display._fig = plt.figure(figsize=(16, 9))
        Display._axes = Display._fig.add_axes((0.1, 0.1, 0.8, 0.8))
        Display._min, Display._max, Display._tmax = 0, 0, 30
        Display._fig.canvas.manager.set_window_title(title)
        Display._axes.set_title(title)
        Display._packs = []
        Display._fig.canvas.mpl_connect("pick_event", Display.picker_hook)

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
                for art in pack["move"][2:]:
                    art.set_zorder(2)
            else:
                for art in pack["move"][2:]:
                    art.set_zorder(1)
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
