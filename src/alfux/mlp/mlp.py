import argparse as arg
import base64
import json
import struct
import sys
import traceback
from typing import Generator, Callable

import numpy as np
from numpy import ndarray
import numpy.random as rng

from .layer import Layer, Neuron
from .processor import Processor


class MLP:
    """Configurable multilayer (or single-layer) perceptron.

    Minimal validation is performed for performance. Use with knowledge of
    expected shapes and types.
    """

    def __init__(self: "MLP", layers: list, cost: Neuron, **kw: dict) -> None:
        """Create a single or multilayer perceptron.

        Args:
            layers (list[Layer]): Hidden layers and output layer.
            cost (Neuron): Cost (loss) function neuron.
            **kw: Optional parameters.

        Keyword Args:
            learning_rate (float): Learning rate. Lower is slower but stable;
                too high may diverge. Defaults to ``1e-3``.
            b1 (float): Adam first moment decay. Defaults to ``0.9``.
            b2 (float): Adam second moment decay. Defaults to ``0.999``.
            preprocess (list[Callable]): Preprocessing pipeline.
            postprocess (list[Callable]): Postprocessing pipeline.
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
        """Return the number of layers in the MLP."""
        return len(self._layers)

    @property
    def preprocess(self: "MLP") -> Callable:
        """Get the compiled preprocessing function."""
        return self._prepro

    @preprocess.setter
    def preprocess(self: "MLP", value: list[Callable]) -> None:
        """Set the preprocessing steps and compile the pipeline."""
        self._save_prepro = value
        self._prepro = Processor.compile_processes(value)

    @property
    def postprocess(self: "MLP") -> Callable:
        """Get the compiled postprocessing function."""
        return self._postpro

    @postprocess.setter
    def postprocess(self: "MLP", value: list[Callable]) -> None:
        """Set the postprocessing steps and compile the pipeline."""
        self._save_postpro = value
        self._postpro = Processor.compile_processes(value)

    @property
    def cost(self: "MLP") -> Neuron:
        """Get the cost (loss) function neuron.

        Returns:
            Neuron: Cost neuron.
        """
        return self._cost

    @property
    def learning_rate(self: "MLP") -> float:
        """Get the learning rate.

        Returns:
            float: Current learning rate.
        """
        return self._lr

    @learning_rate.setter
    def learning_rate(self: "MLP", value: float) -> None:
        """Set the learning rate."""
        self._lr = value

    def eval(self: "MLP", x: ndarray) -> ndarray:
        """Evaluate the MLP output.

        Applies the compiled preprocessing, forward-pass through layers, and
        compiled postprocessing.

        Args:
            x (ndarray): Input data matrix.

        Returns:
            ndarray: Network output.
        """
        x = self._prepro(x)
        for layer in self._layers:
            x = layer.eval(x)
        return self._postpro(x)

    def save(self: "MLP", path: str = "./default.mdl") -> "MLP":
        """Save the MLP to a NumPy file.

        Args:
            path (str): Output file path. Defaults to ``"./default.npy"``.

        Returns:
            MLP: The current instance.
        """
        model = []
        for prepro in self._save_prepro:
            model.append({
                "type": "preprocess",
                "parameters": prepro[1],
                "activation": prepro[0].__name__
            })
        for layer in self._layers:
            model.append({
                "type": "layer",
                "dimension": layer.W.shape,
                "matrix": self.encode_matrix(layer.W),
                "activation": layer.activation
            })
        for postpro in self._save_postpro:
            model.append({
                "type": "postprocess",
                "parameters": postpro[1],
                "activation": postpro[0].__name__
            })
        with open(path, 'w') as file:
            file.write(json.dumps(model, indent=2))
        return self

    def update(self: "MLP", truth: ndarray, data: ndarray) -> None:
        """Perform one epoch of stochastic gradient descent.

        Updates weights using the current cost function and Adam optimizer.

        Args:
            truth (ndarray): Empirical target values (rows are samples).
            data (ndarray): Input samples (rows).
        """
        truth = np.atleast_2d(truth)[:, None, :]
        data = np.atleast_2d(data)[:, None, :]
        for (i, (y, x)) in enumerate(zip(truth, data)):
            print(f"\rPerforming iteration: {i}", end='')
            self._backpropagate(y, np.fromiter(self._forward_pass(x), ndarray))
            self._pb1 *= self._b1
            self._pb2 *= self._b2

    def _backpropagate(self: "MLP", y: ndarray, input: ndarray) -> None:
        """Backpropagate gradients and update internal moments.

        Args:
            y (ndarray): Truth values.
            input (ndarray): Sequence of inputs/outputs per layer, from input
                to final output.
        """
        dk = self._layers[-1].wdiff(input[-2])
        dk = dk @ self._cost.diff(y, input[-1]).T
        dk = np.atleast_1d(dk)
        self._update_layer(-1, np.outer(dk, input[-2]))
        for i in range(len(self._layers) - 2, -1, -1):
            dk = self._layers[i].wdiff(input[i]) @ self._layers[i + 1].W.T @ dk
            self._update_layer(i, np.outer(dk, input[i]))

    def _update_layer(self: "MLP", i: int, gradient: ndarray) -> None:
        """Update a layer's weights using Adam.

        Args:
            i (int): Layer index.
            gradient (ndarray): Gradient for this layer.
        """
        self._m[i] = self._b1 * self._m[i] + (1 - self._b1) * gradient
        self._v[i] = self._b2 * self._v[i] + (1 - self._b2) * gradient ** 2
        m = self._m[i] / (1 - self._pb1)
        v = np.sqrt(self._v[i] / (1 - self._pb2)) + 1e-15
        self._layers[i].W -= self._lr * m / v

    def _forward_pass(self: "MLP", x: ndarray) -> Generator:
        """Yield inputs for each layer including the final output."""
        for layer in self._layers:
            yield x
            x = layer.eval(x)
        yield x

    @staticmethod
    def load(model: list | str) -> "MLP":
        """Load an MLP from a file.

        Args:
            model (list | str): Model description or path to the file.
        Returns:
            MLP: Loaded instance.
        """
        if isinstance(model, str):
            with open(model, "r") as file:
                model = json.loads(file.read())
        arg = {"preprocess": [], "layers": [], "cost": None, "postprocess": []}
        bias = [Neuron("bias")]
        for layer in model:
            match layer["type"]:
                case "preprocess":
                    func = getattr(Processor, layer["activation"])
                    arg["preprocess"].append((func, layer["parameters"]))
                case "postprocess":
                    func = getattr(Processor, layer["activation"])
                    arg["postprocess"].append((func, layer["parameters"]))
                case _:
                    n, m = layer["dimension"]
                    neuron = [Neuron(layer["activation"])]
                    if layer["matrix"] is None:
                        m += 1
                        matrix = np.random.randn(n + 1, m) * np.sqrt(2 / m)
                    else:
                        matrix = MLP.decode_matrix((n, m), layer["matrix"])
                        n -= 1
                    arg["layers"].append(Layer(neuron * n + bias, matrix))
        return MLP(**arg)

    @staticmethod
    def load_legacy(path: str) -> "MLP":
        """Legacy loading for previous versions models.

        Args:
            path (str): Path to the file.
        Returns:
            MLP: Loaded instance.
        """
        return MLP(**np.load(path, allow_pickle=True).item())

    @staticmethod
    def decode_matrix(dim: list, string: str) -> ndarray:
        """Decode a byte string into an ndarray 2d matrix.

        Args:
            dim (list): dimensions of the matrix.
            string (bytes): byte encoded matrix.
        Returns:
            ndarray: The Matrix.
        """
        mat = base64.b64decode(string)
        return np.array(struct.unpack('d' * dim[0] * dim[1], mat)).reshape(dim)

    @staticmethod
    def encode_matrix(mat: ndarray) -> str:
        """Encode a 2d matrix into a byte string.

        Args:
            mat (ndarray): The matrix to encode.
        Returns:
            bytes: The encoded bytestring.
        """
        packed = struct.pack('d' * mat.shape[0] * mat.shape[1], *mat.flatten())
        return base64.b64encode(packed).decode("ascii")


def main() -> int:
    """Sample output test for the MLP.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
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
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
