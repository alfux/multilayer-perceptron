import sys
from typing import Self
from types import MethodType

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
            <neurons> is a Neuron objects list used as neurons in the layer.
            <weights> is a matrix with lines equal to the number of neurons and
            columns equal to the number of input values.
        """
        self._matrix: ndarray = weights
        match len(neurons):
            case 0:
                raise ValueError("neurons can't be empty")
            case 1:
                self._neurons: Neuron = neurons[0]
                self.__class__.__call__ = MethodType(Layer._vect_call, self)
                self.wdiff = MethodType(Layer._vect_wdiff, self)
            case _:
                if len(neurons) != len(weights):
                    raise ValueError("matrix' shape doesn't fit neuron list")
                self._neurons: list[Neuron] = neurons
                self.__class__.__call__ = MethodType(Layer._call, self)
                self.wdiff = MethodType(Layer._wdiff, self)

    def __len__(self: Self) -> int:
        """Returns the length of the layer."""
        return len(self._matrix)

    def __repr__(self: Self) -> str:
        """String representation of the object."""
        string = f"{self._matrix.shape}"
        if isinstance(self._neurons, list):
            for neuron in self._neurons:
                string += f"\n\t{neuron}"
        else:
            string += f"\n\t{self._neurons}"
        return string

    def _vect_call(self: Self, input: ndarray) -> ndarray:
        """Computes layer's weighted output as a single-neural layer.

        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        return self._neurons(input)

    def _vect_wdiff(self: Self, input: ndarray) -> ndarray:
        """Weighted derivative function of the layer. Computes differential in
        <self._weigths> @ <input> as a single-neural layer.

        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        return self._neurons.diff(input)

    def _call(self: Self, input: ndarray) -> ndarray:
        """Computes layer's weighted output as a multi-neural layer.

        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        out = np.empty(len(self._matrix), float)
        for i in range(len(self._matrix)):
            out[i] = self._neurons[i](input[i])
        return out

    def _wdiff(self: Self, input: ndarray) -> ndarray:
        """Weighted derivative function of the layer. Computes differential in
        <self._weigths> @ <input> as a multi-neural layer.

        By construction of the layer, the differential is a diagonal matrix.
        Args:
            If weights are a (l, m) matrix, <input> is expected to have
            length m.
        """
        input = self._matrix @ input
        out = np.zeros((len(self._matrix), len(self._matrix)), float)
        for i in range(len(self._matrix)):
            out[i, i] = self._neurons[i].diff(input[i])
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
        layer = Layer(Neuron(funct, deriv), matrix, True)
        print(f"\nlayer({vectr.T}) = \n\n{layer(vectr)}\n")
        print(f"\nlayer.deriv({vectr.T}) = \n\n{layer.wdiff(vectr)}\n")
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
