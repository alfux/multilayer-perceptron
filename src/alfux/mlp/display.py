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
            Display._fig = plt.figure(figsize=(16, 9))
        Display._fig.canvas.manager.set_window_title(title)
        if Display._axes is None:
            Display._axes = Display._fig.add_axes((0.1, 0.1, 0.8, 0.8))
        Display._axes.set_title(title)
        self._start = time.time()
        self._margin = margin
        self._times = [[] for _ in range(n)]
        self._values = [[] for _ in range(n)]
        self._min, self._max = 0, 0
        if len(param) < n:
            param += [{}] * (n - len(param))
        self._lines = [
            self._axes.plot(t, v, **p)[0]
            for t, v, p in zip(self._times, self._values, param)
        ]
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
        if value < self._min:
            self._min = value
        elif value > self._max:
            self._max = value
        self._axes.set_ylim(self._min - self._margin, self._max + self._margin)
        self._axes.set_xlim(0, np.max([30, self._times[i][-1]]))
        self._lines[i].set_data(self._times[i], self._values[i])
        plt.draw()
        plt.pause(1e-15)

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
