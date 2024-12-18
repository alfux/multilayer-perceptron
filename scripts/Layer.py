import sys
from typing import Self

import numpy as np
from numpy import ndarray

from Neuron import Neuron


class Layer:
    """Layer of a neural network."""

    def __init__(self: Self, neurons: list, weights: ndarray) -> None:
        """Creates the layer based on neurons list and weights matrix."""
        self._neurons: Neuron = neurons
        self._matrix: ndarray = weights

    def __call__(self: Self, vec: ndarray) -> ndarray:
        """Computes layer's output."""
        vec = self._matrix @ vec.T
        out = np.empty(len(self._neurons), float)
        for i in range(len(self._neurons)):
            out[i] = self._neurons[i](vec[i])
        return out

    def deriv(self: Self, vec: ndarray) -> ndarray:
        """Computes layer's differential"""
        vec = self._matrix @ vec.T
        out = np.empty(len(self._neurons), float)
        for i in range(len(self._neurons)):
            out[i] = self._neurons[i].deriv(vec[i])
        return out

    @property
    def weights(self: Self) -> ndarray:
        """Returns the matrix of weights."""
        return self._matrix

    @weights.setter
    def weights(self: Self, value: ndarray) -> None:
        self._matrix = value


def main() -> int:
    """Displays the output of a layer."""
    try:
        print("Layer presentation:")
        end = False
        neurons = []
        vector = []
        while not end:
            try:
                funct = eval(input("\tfunct: "))
                deriv = eval(input("\tderiv: "))
                vector.append(int(input("\tvalue: ")))
                neurons.append(Neuron(funct, deriv))
            except Exception as err:
                print(f"Error: {type(err).__name__}: {err}", file=sys.stderr)
            finally:
                end = input("Continue ? (y/n): ").casefold() in ('n', "no")
        matrix = np.random.rand(len(neurons), len(vector))
        print(f"Matrix is\n{matrix}")
        layer = Layer(neurons, matrix)
        print(f"\nlayer({vector}) = {layer(np.array(vector))}")
        print(f"\nlayer.deriv({vector}) = {layer.deriv(np.array(vector))}")
        return 0
    except Exception as err:
        print(f"Fatal: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
