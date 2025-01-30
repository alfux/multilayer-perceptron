import argparse as arg
import sys
import traceback
from typing import Self, Generator, Callable

import numpy as np
from numpy import ndarray
import numpy.random as rng

from Layer import Layer
from Neuron import Neuron


class MLP:
    """Configurable Multilayer (or Singlelayer) Perceptron

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, layers: list, cost: Neuron, **kw: dict) -> None:
        """Creates a single or multilayer perceptron.

        Args:
            <layers> is a list of Layer objects. They represent hidden layers
            and the output layer of the network.
            <cost> is the cost function used to train the model and measure
            its performance.
        Keyword arguments <kw>:
            <learning_rate> is the speed of learning, the lower the learning
            rate, the slower the convergence. But a higher learning rate may
            lead complete to divergence.
            <preprocess> is the preprocessing function used over the training
            dataset. It is used by the model to have consistent input.
        """
        self._layers: list[Layer] = layers
        self._cost: Neuron = cost
        self._lr: float = kw.get("learning_rate", 1e-3)
        self._b1: float = kw.get("b1", 0.9)
        self._pb1: float = self._b1
        self._m: list = [np.zeros(layer.W.shape) for layer in layers]
        self._b2: float = kw.get("b2", 0.999)
        self._pb2: float = self._b2
        self._v: list = [np.zeros(layer.W.shape) for layer in layers]
        self.preprocess: Callable = kw.get("preprocess", None)

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

    @property
    def preprocess(self: Self) -> Callable[[ndarray], ndarray]:
        return self._preprocess

    @preprocess.setter
    def preprocess(self: Self, value: Callable[[ndarray], ndarray]) -> None:
        self._preprocess = value
        if value is not None:
            self.eval = self._preprocessed_eval
        else:
            self.eval = self._vanilla_eval

    @property
    def cost(self: Self) -> Neuron:
        """Getter for the cost function."""
        return self._cost

    @property
    def learning_rate(self: Self) -> float:
        """Getter for the learning_rate."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self: Self, value: float) -> None:
        """Setter for the learning_rate."""
        self._lr = value

    def eval(self: Self, x: ndarray) -> ndarray:
        """Evaluates the MLP's output.

        This function is dynamically allocated depending on the presence of a
        preprocess function or not.
        Args:
            <x> is the input of the layer.
        Returns:
            The output of the last layer.
        """
        pass

    def save(self: Self, path: str) -> None:
        """Saves the current state of the MLP into a file.

        Args:
            <path> is the location and name of the save file.
        Returns:
            None
        """
        # Unimplemented yet.
        pass

    def update(self: Self, truth: ndarray, data: ndarray) -> None:
        """Updates the model by one pass of stochastic gradient descent.

        The update is based on the model's current cost function.
        Args:
            <data> is the matrix containing, by row, inputs to train over.
        """
        for (y, x) in zip(truth, data):
            self._backpropagate(y, np.fromiter(self._forward_pass(x), ndarray))
            self._pb1 *= self._b1
            self._pb2 *= self._b2

    def _backpropagate(self: Self, y: ndarray, input: ndarray) -> None:
        """Updates matrices with backpropagation.

        Args:
            <input> is the chain of input / output in the network.
            First element is the initial input.
            Last element is the last layer's output.
            <y> is the truth values to compare against in the loss function.
        Returns:
            None
        """
        dk = self._layers[-1].wdiff(input[-2]) @ self._cost.diff(y, input[-1])
        self._update_layer(-1, np.outer(dk, input[-2]))
        for i in range(len(self._layers) - 2, -1, -1):
            dk = self._layers[i].wdiff(input[i]) @ self._layers[i + 1].W.T @ dk
            self._update_layer(i, np.outer(dk, input[i]))

    def _update_layer(self: Self, i: int, gradient: ndarray) -> None:
        """Updates layer with ADAM momentum.

        Args:
            <i> index of the layer.
            <gradient> is the gradient of the composition from loss function up
            until layer <i>.
        Returns:
            None
        """
        self._m[i] = self._b1 * self._m[i] + (1 - self._b1) * gradient
        self._v[i] = self._b2 * self._v[i] + (1 - self._b2) * gradient ** 2
        self._layers[i].W -= (
            self._lr * (self._m[i] / (1 - self._pb1))
            / (np.sqrt(self._v[i] / (1 - self._pb2)) + 1e-15)
        )

    def _forward_pass(self: Self, x: ndarray) -> Generator:
        """Generates an array with inputs of each layer (cost included)."""
        for layer in self._layers:
            yield x
            x = layer.eval(x)
        yield x

    def _preprocessed_eval(self: Self, x: ndarray) -> ndarray:
        """Computes the network's output

        Args:
            <x> is supposed to be a (m, n) matrix where m is the number of
            entries of the input layer plus one.
        """
        self._preprocess(x)
        for layer in self._layers:
            x = layer.eval(x)
        return x

    def _vanilla_eval(self: Self, x: ndarray) -> ndarray:
        """Computes the network's output

        Args:
            <x> is supposed to be a (m, n) matrix where m is the number of
            entries of the input layer plus one.
        """
        for layer in self._layers:
            x = layer.eval(x)
        return x

    @staticmethod
    def load(string: str, direct: bool = False) -> Self:
        """Loads MLP parameters from a file or a string."""
        if direct:
            return  # load_logic(string)
        with open(string, 'rb') as file:
            return  # load_logic(file.read())


def main() -> int:
    """MLP sample output test."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("m", help="number of first layer's input", type=int)
        av.add_argument("n", help="number of second layer's input", type=int)
        av.add_argument("o", help="number of second layer's output", type=int)
        av.add_argument("f", help="function used in each layer", type=str)
        av.add_argument("df", help="derivative of the function", type=str)
        av = av.parse_args()
        (f, df) = (eval(av.f), eval(av.df))
        l1 = Layer([Neuron(f, df)] * av.n, np.round(rng.rand(av.n, av.m)))
        l2 = Layer([Neuron(f, df)] * av.o, np.round(rng.rand(av.o, av.n)))
        cost = Neuron(lambda x: 2 * x, lambda x: 2 * np.identity(av.o))
        (mlp, x) = (MLP([l1, l2], cost), np.array(eval(input("Input: "))))
        print(f"\nl1.W = \n\n{l1.W}\n\nl1({x}) = \n\n\t{l1.eval(x)}")
        print(f"\nl2.W = \n\n{l2.W}\n\nl2({l1.eval(x)})", end='')
        print(f"= \n\n\t{l2.eval(l1.eval(x))}\nmlp({x}) = {mlp.eval(x)}")
        print(f"\ncost(mlp({x})) = {mlp.cost.eval(mlp.eval(x))}")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
