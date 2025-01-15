import sys
from typing import Self

import numpy as np
from numpy import ndarray

from Neuron import Neuron


class Layer:
    """Layer of neurons of a neural network. It is a vector of neurons.

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside of __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, neuron: Neuron, weights: ndarray) -> None:
        """Creates the layer based on neurons list and weights matrix.

        Args:
            <neurons> is a Neuron objects used as basis in the layer.
            <weights> is a matrix with lines equal to the number of neurons and
            columns equal to the number of input values.
        """
        self._neuron: Neuron = neuron
        self._matrix: ndarray = weights

    def __len__(self: Self) -> int:
        """Returns the length of the layer."""
        return len(self._matrix)

    def __call__(self: Self, input: ndarray) -> ndarray:
        """Computes layer's weighted output.

        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        out = np.empty(len(self._matrix), float)
        for i in range(len(self._matrix)):
            out[i] = self._neuron(input[i])
        return out

    def wdiff(self: Self, input: ndarray) -> ndarray:
        """Weighted derivative function of the layer. Computes differential in
        <self._weigths> @ <input>.

        By construction of the layer, the differential is a diagonal matrix.
        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        out = np.zeros((len(self._matrix), len(self._matrix)), float)
        for i in range(len(self._matrix)):
            out[i, i] = self._neuron.diff(input[i])
        return out

    @property
    def W(self: Self) -> ndarray:
        """Get the matrix of weights."""
        return self._matrix

    @W.setter
    def W(self: Self, value: ndarray) -> None:
        """Set the matrix of weights."""
        self._matrix = value


def main() -> int:
    """Displays the output of a layer."""
    try:
        print("Layer presentation:\n")
        funct = eval(input("\tNeuron's function: "))
        deriv = eval(input("\tNeuron's derivative: "))
        numbr = int(input("\tNumber of neurons: "))
        vectr = np.array(eval(input("\tInputs: ")))
        matrix = np.random.rand(numbr, len(vectr))
        print(f"Matrix is\n\n{matrix}")
        layer = Layer(Neuron(funct, deriv), matrix)
        print(f"\nlayer({vectr.T}) = \n\n{layer(vectr)}\n")
        print(f"\nlayer.deriv({vectr.T}) = \n\n{layer.wdiff(vectr)}\n")
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
