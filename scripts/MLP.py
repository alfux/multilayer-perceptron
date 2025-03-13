import argparse as arg
import sys
import traceback
from typing import Generator, Callable

import numpy as np
from numpy import ndarray
import numpy.random as rng

from Layer import Layer, Neuron
from Processor import Processor


class MLP:
    """Configurable Multilayer (or Singlelayer) Perceptron

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: "MLP", layers: list, cost: Neuron, **kw: dict) -> None:
        """Creates a single or multilayer perceptron.

        Args:
            layers is a list of Layer objects. They represent hidden layers
            and the output layer of the network.
            cost is the cost function used to train the model and measure
            its performance.
        Keyword arguments <kw>:
            learning_rate is the speed of learning, the lower the learning
            rate, the slower the convergence. But a higher learning rate may
            lead complete to divergence.
            preprocess is the preprocessing list of functions used over the
            training dataset.
            postprocess is the postprocessing list of functions used over the
            training output.
        """
        self._layers: list[Layer] = layers
        self._cost: Neuron = cost
        self._lr: float = kw.get("learning_rate", 1e-3)
        (self._b1, self._b2) = (kw.get("b1", 0.9), kw.get("b2", 0.999))
        (self._pb1, self._pb2) = (self._b1, self._b2)
        self._m: list = [np.zeros(layer.W.shape) for layer in layers]
        self._v: list = [np.zeros(layer.W.shape) for layer in layers]
        self.preprocess = kw.get("preprocess", [])
        self.postprocess = kw.get("postprocess", [])

    def __len__(self: "MLP") -> int:
        """Returns the number of layers in the MLP."""
        return len(self._layers)

    @property
    def preprocess(self: "MLP") -> Callable:
        """Getter for the preprocessor."""
        return self._prepro

    @preprocess.setter
    def preprocess(self: "MLP", value: list[Callable]) -> None:
        """Setter for the preprocessor."""
        self._save_prepro = value
        self._prepro = Processor.compile_processes(value)

    @property
    def postprocess(self: "MLP") -> Callable:
        """Getter for the postprocessor."""
        return self._postpro

    @postprocess.setter
    def postprocess(self: "MLP", value: list[Callable]) -> None:
        """Setter for the postprocessor."""
        self._save_postpro = value
        self._postpro = Processor.compile_processes(value)

    @property
    def cost(self: "MLP") -> Neuron:
        """Getter for the cost function."""
        return self._cost

    @property
    def learning_rate(self: "MLP") -> float:
        """Getter for the learning_rate."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self: "MLP", value: float) -> None:
        """Setter for the learning_rate."""
        self._lr = value

    def eval(self: "MLP", x: ndarray) -> ndarray:
        """Evaluates the MLP's output.

        This function is dynamically allocated depending on the presence of a
        preprocess function or not.
        Args:
            x is the input of the layer.
        Returns:
            The output of the last layer.
        """
        x = self._prepro(x)
        for layer in self._layers:
            x = layer.eval(x)
        return self._postpro(x)

    def save(self: "MLP", path: str = "./default.npy") -> "MLP":
        """Saves the MLP in a numpy file.

        Args:
            path is the save file path
        Returns:
            The current instance
        """
        mlp = {
            "layers": self._layers, "cost": self._cost,
            "learning_rate": self._lr, "b1": self._b1, "b2": self._b2,
            "preprocess": self._save_prepro, "postprocess": self._save_postpro
        }
        np.save(path, mlp, allow_pickle=True)
        return self

    def update(self: "MLP", truth: ndarray, data: ndarray) -> None:
        """Updates the model by one pass of stochastic gradient descent.

        The update is based on the model's current cost function.
        Args:
            <data> is the matrix containing, by row, inputs to train over.
        """
        for (i, (y, x)) in enumerate(zip(truth, data)):
            print(f"\rPerforming iteration: {i}", end='')
            self._backpropagate(y, np.fromiter(self._forward_pass(x), ndarray))
            self._pb1 *= self._b1
            self._pb2 *= self._b2

    def _backpropagate(self: "MLP", y: ndarray, input: ndarray) -> None:
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
        dk = np.atleast_1d(dk)
        self._update_layer(-1, np.outer(dk, input[-2]))
        for i in range(len(self._layers) - 2, -1, -1):
            dk = self._layers[i].wdiff(input[i]) @ self._layers[i + 1].W.T @ dk
            self._update_layer(i, np.outer(dk, input[i]))

    def _update_layer(self: "MLP", i: int, gradient: ndarray) -> None:
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
        m = self._m[i] / (1 - self._pb1)
        v = np.sqrt(self._v[i] / (1 - self._pb2)) + 1e-15
        self._layers[i].W -= self._lr * m / v

    def _forward_pass(self: "MLP", x: ndarray) -> Generator:
        """Generates an array with inputs of each layer (cost included)."""
        for layer in self._layers:
            yield x
            x = layer.eval(x)
        yield x

    @staticmethod
    def load(path: str) -> "MLP":
        """Loads an MLP

        Args:
            path is the path to the file to load
        Returns:
            The loaded instance of the MLP
        """
        return MLP(**np.load(path, allow_pickle=True).item())


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
        A = Layer([Neuron(av.f, av.df)] * av.n, np.round(rng.rand(av.n, av.m)))
        B = Layer([Neuron(av.f, av.df)] * av.o, np.round(rng.rand(av.o, av.n)))
        cost = Neuron("lambda x: 2 * x", f"lambda x: 2 * {np.identity(av.o)}")
        (mlp, x) = (MLP([A, B], cost), np.array(eval(input("Input: "))))
        mlp = eval(repr(mlp))
        print(f"\nl1.W = \n\n{A.W}\n\nl1({x}) = \n\n\t{A.eval(x)}")
        print(f"\nl2.W = \n\n{B.W}\n\nl2({A.eval(x)})", end='')
        print(f"= \n\n\t{B.eval(A.eval(x))}\nmlp({x}) = {mlp.eval(x)}")
        print(f"\ncost(mlp({x})) = {mlp.cost.eval(mlp.eval(x))}")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
