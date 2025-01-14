import sys
from typing import Self, Generator
import traceback

import numpy as np
from numpy import ndarray

from Neuron import Neuron
from Layer import Layer


class MultilayerPerceptron:
    """Configurable Multilayer (or Singlelayer) Perceptron

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, layers: list[Layer], **kwargs: dict) -> None:
        """Creates a single or multilayer perceptron.

        Args:
            <layers> is a list of Layer objects. They represent hidden layers
            and the output layer of the network.
        """
        self._layers: list[Layer] = layers
        self._cost: Neuron = kwargs["cost"] if "cost" in kwargs else None
        self._lr: float = kwargs["lrate"] if "lrate" in kwargs else 1e-3

    def __call__(self: Self, vec: ndarray) -> float:
        """Computes the network's output

        Args:
            <vec> is supposed to be a (m, n) matrix where m is the number of
            entries of the input layer plus one.
        """
        for layer in self._layers:
            vec = layer(vec)
        return vec

    def update(self: Self, data: ndarray) -> None:
        """Updates the model by one pass of stochastic gradient descent.
        
        The update is based on the model's current cost function.
        Args:
            <data> is the matrix containing, by row, inputs to train over.
        """
        for row in data:
            self._backpropagate(np.fromiter(self._forward_pass(row), ndarray))

    def _backpropagate(self: Self, input: ndarray) -> None:
        """Updates matrices with backpropagation.

        Args:
            <input> is the chain of input / output in the network.
            First element is the initial input.
            Last element is the last layer's output.
        """
        delta = (
            self._layers[-1].diff(self._layers[-1].weights @ input[-2]) @
            self._cost.diff(input[-1]).T
        )
        self._layers[-1].weights = (
            self._layers[-1].weights -
            self._lr * np.outer(delta, input[-2])
        )
        for i in range(len(self._layers) - 2, -1, -1):
            delta =  (
                self._layers[i].diff(self._layers[i].weights @ input[i]) @
                self._layers[i + 1].weights.T @ delta
            )
            self._layers[i].weights = (
                self._layers[i].weights -
                self._lr * np.outer(delta, input[i])
            )

    def _forward_pass(self: Self, row: ndarray) -> Generator:
        """Generates an array with inputs of each layer (cost included)."""
        for layer in self._layers:
            yield row
            row = layer(row)
        yield row

def main() -> int:
    try:
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    main()
