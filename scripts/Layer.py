import argparse as arg
import sys
import traceback
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
                self.eval = self._vect_eval
                self.wdiff = self._vect_wdiff
            case _:
                if len(neurons) != len(weights):
                    print(weights.shape)
                    print(len(neurons))
                    raise ValueError("matrix' shape doesn't fit neuron list")
                self._neurons: list[Neuron] = neurons
                self.eval = self._eval
                self.wdiff = self._wdiff

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

    @property
    def W(self: Self) -> ndarray:
        """Get the matrix of weights."""
        return self._matrix

    @W.setter
    def W(self: Self, value: ndarray) -> None:
        """Set the matrix of weights."""
        self._matrix = value

    def eval(self: Self, x: ndarray) -> ndarray:
        """Computes layer's weighted output.

        Args:
            <x> is the input vector. It is first multiplied by <self._matrix>.
        """
        pass

    def _vect_eval(self: Self, x: ndarray) -> ndarray:
        """Computes layer's weighted output as a single-neural layer.

        Args:
            If weights are a (l, m) matrix, <x> is expected to have
            length m.
        """
        x = (self._matrix @ x.T).T
        return self._neurons.eval(x)

    def _vect_wdiff(self: Self, x: ndarray) -> ndarray:
        """Weighted derivative function of the layer. Computes differential in
        <self._weigths> @ <x> as a single-neural layer.

        Args:
            If weights are a (l, m) matrix, <x> is expected to have
            length m.
        """
        x = (self._matrix @ x.T).T
        return self._neurons.diff(x)

    def _eval(self: Self, x: ndarray) -> ndarray:
        """Computes layer's weighted output as a multi-neural layer.

        Args:
            If weights are a (l, m) matrix, <x> is expected to have
            length m.
        """
        x = self._matrix @ x.T
        for i in range(x.shape[0]):
            x[i] = self._neurons[i].eval(x[i])
        return x.T

    def _wdiff(self: Self, x: ndarray) -> ndarray:
        """Weighted derivative function of the layer. Computes differential in
        <self._weigths> @ <x> as a multi-neural layer.

        By construction of the layer, the differential is a diagonal matrix.
        Args:
            If weights are a (l, m) matrix, <x> is expected to have
            length m.
        """
        x = self._matrix @ x
        out = np.zeros((len(self._matrix), len(self._matrix)), float)
        for i in range(len(self._matrix)):
            out[i, i] = self._neurons[i].diff(x[i])
        return out


def main() -> int:
    """Displays the output of a layer."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av = av.parse_args()
        print("Layer presentation:\n")
        funct = eval(input("\tNeuron's function: "))
        deriv = eval(input("\tNeuron's derivative: "))
        numbr = int(input("\tNumber of neurons: "))
        vectr = np.array(eval(input("\tInputs: ")))
        matrix = np.random.rand(numbr, len(vectr))
        print(f"Matrix is\n\n{matrix}")
        layer = Layer([Neuron(funct, deriv)] * numbr, matrix)
        print(f"\nlayer({vectr.T}) = \n\n{layer.eval(vectr)}\n")
        print(f"\nlayer.deriv({vectr.T}) = \n\n{layer.wdiff(vectr)}\n")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
