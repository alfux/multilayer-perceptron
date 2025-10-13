import argparse as arg
import sys
import traceback
from typing import Self

import numpy as np
from numpy import ndarray

from .neuron import Neuron


class Layer:
    """Layer of neurons in a neural network.

    Represents a vector of neurons with an associated weight matrix. This
    class assumes correct inputs and does minimal validation to reduce
    complexity and overhead.
    """

    def __init__(self: Self, neurons: list[Neuron], weights: ndarray) -> None:
        """Initialize a layer with neurons and weights.

        Args:
            neurons (list[Neuron]): Neurons in the layer.
            weights (ndarray): Weight matrix shaped ``(n_neurons, n_inputs)``.
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
                    print(weights.shape, len(neurons), sep="\n")
                    raise ValueError("matrix' shape doesn't fit neuron list")
                self._neurons: list[Neuron] = neurons
                self.eval = self._eval
                self.wdiff = self._wdiff
        self._activation = self._neurons[0].activation

    def __len__(self: Self) -> int:
        """Return the number of neurons in the layer."""
        return len(self._matrix)

    @property
    def activation(self: Self) -> str:
        """str: Get the activation function's name."""
        return self._activation

    @property
    def W(self: Self) -> ndarray:
        """Get the weight matrix."""
        return self._matrix

    @W.setter
    def W(self: Self, value: ndarray) -> None:
        """Set the weight matrix."""
        self._matrix = value

    def eval(self: Self, x: ndarray) -> ndarray:
        """Compute the layer output.

        Args:
            x (ndarray): Input vector or batch. Multiplied by the weight
                matrix then passed through neuron activations.
        """
        pass

    def _vect_eval(self: Self, x: ndarray) -> ndarray:
        """Compute output for a single-neuron layer.

        Args:
            x (ndarray): Input of length ``m`` for weights shaped ``(l, m)``.

        Returns:
            ndarray: Activated output.
        """
        x = (self._matrix @ x.T).T
        return self._neurons.eval(x)

    def _vect_wdiff(self: Self, x: ndarray) -> ndarray:
        """Compute weighted derivative for a single-neuron layer.

        Args:
            x (ndarray): Input of length ``m`` for weights shaped ``(l, m)``.

        Returns:
            ndarray: Derivative evaluated at ``W @ x``.
        """
        x = (self._matrix @ x.T).T
        return self._neurons.diff(x)

    def _eval(self: Self, x: ndarray) -> ndarray:
        """Compute output for a multi-neuron layer.

        Args:
            x (ndarray): Input of length ``m`` for weights shaped ``(l, m)``.

        Returns:
            ndarray: Activated output with shape ``(batch, l)``.
        """
        x = self._matrix @ x.T
        for i in range(x.shape[0]):
            x[i] = self._neurons[i].eval(x[i])
        return x.T

    def _wdiff(self: Self, x: ndarray) -> ndarray:
        """Compute weighted derivative for a multi-neuron layer.

        Computes the derivative of ``W @ x`` evaluated by each neuron's
        derivative. By construction, the result is a diagonal matrix.

        Args:
            x (ndarray): Input of length ``m`` for weights shaped ``(l, m)``.

        Returns:
            ndarray: Diagonal matrix of derivatives with shape ``(l, l)``.
        """
        x = self._matrix @ x.T
        out = np.zeros((len(self._matrix), len(self._matrix)), float)
        for i in range(len(self._matrix)):
            out[i, i] = self._neurons[i].diff(x[i])
        return out


def main() -> int:
    """Display an example layer output.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av = av.parse_args()
        print("Layer presentation:\n")
        funct = input("\tNeuron's function: ")
        deriv = input("\tNeuron's derivative: ")
        numbr = int(input("\tNumber of neurons: "))
        vectr = np.array(eval(input("\tInputs: ")))
        matrix = np.random.rand(numbr, np.size(vectr))
        print(f"Matrix is\n\n{matrix}")
        layer = Layer([Neuron(funct, deriv)] * numbr, matrix)
        print(f"\nlayer({vectr.T}) = \n\n{layer.eval(np.atleast_1d(vectr))}\n")
        print(f"\nlayer.deriv({vectr.T}) = \n")
        print(f"{layer.wdiff(np.atleast_1d(vectr))}\n")
        print("Representation:", layer, sep="\n")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
