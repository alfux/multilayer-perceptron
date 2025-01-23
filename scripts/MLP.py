import argparse as arg
import sys
from typing import Self, Generator, Callable

import numpy as np
from numpy import ndarray

from Neuron import Neuron
from Layer import Layer


class MLP:
    """Configurable Multilayer (or Singlelayer) Perceptron

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, layers: list, cost: Neuron, lr=1e-3) -> None:
        """Creates a single or multilayer perceptron.

        Args:
            <layers> is a list of Layer objects. They represent hidden layers
            and the output layer of the network.
            <cost> is the cost function used to train the model and measure
            its performance.
            <lr> is the learning rate. The lower the learning rate, the slower
            the convergence. But a higher learning rate may lead to divergence.
        """
        self._layers: list[Layer] = layers
        self._cost: Neuron = cost
        self._lr: float = lr

    def __call__(self: Self, vec: ndarray) -> float:
        """Computes the network's output

        Args:
            <vec> is supposed to be a (m, n) matrix where m is the number of
            entries of the input layer plus one.
        """
        for layer in self._layers:
            vec = layer(vec)
        return vec

    def __len__(self: Self) -> int:
        """Returns the number of layers in the MLP."""
        return len(self._layers)

    def __repr__(self: Self) -> str:
        """String representation of the object."""
        string = ""
        for layer in self._layers:
            string += f"\n{layer}"
        string += f"\n{self._cost}"
        return string

    def update(self: Self, data: ndarray) -> None:
        """Updates the model by one pass of stochastic gradient descent.

        The update is based on the model's current cost function.
        Args:
            <data> is the matrix containing, by row, inputs to train over.
        """
        for row in data:
            self._backpropagate(np.fromiter(self._forward_pass(row), ndarray))

    @property
    def cost(self: Self) -> Callable:
        """Getter for the cost function."""
        return self._cost

    def _backpropagate(self: Self, input: ndarray) -> None:
        """Updates matrices with backpropagation.

        Args:
            <input> is the chain of input / output in the network.
            First element is the initial input.
            Last element is the last layer's output.
            It is computed during a forward pass
        """
        dk = self._layers[-1].wdiff(input[-2]) @ self._cost.diff(input[-1])
        self._layers[-1].W -= self._lr * np.outer(dk, input[-2])
        for i in range(len(self._layers) - 2, -1, -1):
            dk = self._layers[i].wdiff(input[i]) @ self._layers[i + 1].W.T @ dk
            self._layers[i].W -= self._lr * np.outer(dk, input[i])

    def _forward_pass(self: Self, vec: ndarray) -> Generator:
        """Generates an array with inputs of each layer (cost included)."""
        for layer in self._layers:
            yield vec
            vec = layer(vec)
        yield vec


def main() -> int:
    """MLP sample output test."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("m", help="number of first layer's input", type=int)
        av.add_argument("n", help="number of second layer's input", type=int)
        av.add_argument("o", help="number of second layer's output", type=int)
        av.add_argument("f", help="function used in each layer", type=str)
        av.add_argument("df", help="derivative of the function", type=str)
        av = av.parse_args()
        print("MultiLayerPerceptron example")
        (f, df) = (eval(av.f), eval(av.df))
        l1 = Layer(Neuron(f, df), np.round(np.random.rand(av.n, av.m)))
        l2 = Layer(Neuron(f, df), np.round(np.random.rand(av.o, av.n)))
        cost = Neuron(lambda x: 2 * x, lambda x: 2 * np.identity(av.o))
        mlp = MLP([l1, l2], cost)
        x = eval(input("Input vector: "))
        print(f"\nl1.W = \n\n{l1.W}\n\nl1({x}) = \n\n\t{l1(x)}")
        print(f"\nl2.W = \n\n{l2.W}\n\nl2({l1(x)}) = \n\n\t{l2(l1(x))}")
        print(f"\nmlp({x}) = {mlp(x)}\n\ncost(mlp({x})) = {mlp.cost(mlp(x))}")
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
