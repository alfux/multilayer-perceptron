import sys
from typing import Self
import traceback

import numpy as np
from numpy import ndarray

from Neuron import Neuron


class Layer:
    """Layer of neurons of a neural network. It is a vector of neurons.

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside of __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, neurons: list[Neuron], weights: ndarray) -> None:
        """Creates the layer based on neurons list and weights matrix.

        Args:
            <neurons> is a list of Neuron objects.
            <weights> is a matrix with lines equal to the number of neurons and
            columns equal to the number of input values plus one for the bias.
        """
        self._neurons: list[Neuron] = neurons
        self._matrix: ndarray = weights

    def __call__(self: Self, vec: ndarray) -> ndarray:
        """Computes layer's output.

        Args:
            If weights are a (l, m) matrix, <vec> is expected to have length m.
        """
        vec = self._matrix @ vec
        out = np.zeros(len(self._neurons), float)
        for i in range(len(self._neurons)):
            out[i] = self._neurons[i](vec[i])
        return out

    def diff(self: Self, vec: ndarray) -> ndarray:
        """Derivative function of the layer. Computes differential in <vec>.

        By construction of the layer, the differential is a diagonal matrix.
        Args:
            If weights are a (l, m) matrix, <vec> is expected to have length m.
        """
        vec = self._matrix @ vec
        out = np.empty((len(self._neurons), len(self._neurons)), float)
        for i in range(len(self._neurons)):
            out[i, i] = self._neurons[i].diff(vec[i])
        return out

    @property
    def weights(self: Self) -> ndarray:
        """Get the matrix of weights."""
        return self._matrix

    @weights.setter
    def weights(self: Self, value: ndarray) -> None:
        """Set the matrix of weights."""
        self._matrix = value


def main() -> int:
    """Displays the output of a layer."""
    try:
        print("Layer presentation:")
        (neurons, vector, end) = ([], [], False)
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
        vector = np.atleast_2d(np.array(vector)).T
        print(f"\nlayer({vector.T}) = \n{layer(vector)}")
        print(f"\nlayer.deriv({vector.T}) = \n{layer.diff(vector)}")
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
