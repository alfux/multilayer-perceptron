import argparse as arg
import sys
from typing import Self, Callable

import pandas as pd
import numpy as np
from numpy import ndarray


class Neuron:
    """Neuron node for a neural network."""

    def __init__(self: Self, weights: ndarray, activation: Callable) -> None:
        """Define neuron with its weights and activation function."""
        self._weights = weights
        self._activation = activation

    @property
    def weights(self: Self) -> ndarray:
        """Returns weights of the neuron, including bias at index 0."""
        return self._weights

    @weights.setter
    def weights(self: Self, new_weights: ndarray) -> None:
        """Updates neuron's weights."""
        if not isinstance(new_weights, ndarray):
            raise TypeError("weights must be ndarray")
        if new_weights.ndim != 1:
            raise ValueError("weights must be 1D-array")
        self._weights = new_weights
    
    def __call__(self: Self, data: ndarray) -> float:
        """Computes neuron output."""
        if not isinstance(data, ndarray):
            raise TypeError("data must be ndarray")
        if data.shape[0] == 0:
            raise ValueError("data is empty")
        if data.ndim == 1:
            data = data.reshape((1, -1))
        if data.shape[1] != len(self._weights) - 1:
            raise ValueError("data's columns should match number of weights")
        if data.ndim != 2:
            raise ValueError("data's dimension should be 1 or 2")
        bias_data = np.hstack((np.ones((data.shape[0], 1)), data))
        return self._activation(np.dot(bias_data, self._weights))


def main() -> None:
    """Displays neuron output from dataset input."""
    try:
        parser = arg.ArgumentParser(description=main.__doc__)
        parser.add_argument("data", help="csv dataset to compute")
        parser.add_argument("weights", help="csv weights to use")
        data = pd.read_csv(parser.parse_args().data)
        weights = pd.read_csv(parser.parse_args().weights)
        print(data, weights, sep="\n\n")
    except Exception as err:
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
