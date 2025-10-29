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
        self._last_matrices: list[ndarray] = [x.W for x in self._layers]
        self._cost: Neuron = cost
        self._lr: float = kw.get("learning_rate", 1e-3)
        (self._b1, self._b2) = (kw.get("b1", 0.9), kw.get("b2", 0.999))
        (self._pb1, self._pb2) = (self._b1, self._b2)
        self._m: list = [np.zeros(layer.W.shape) for layer in layers]
        self._v: list = [np.zeros(layer.W.shape) for layer in layers]
        self.preprocess = kw.get("preprocess", [])
        self.postprocess = kw.get("postprocess", [])
        self._last_gradient = None
        if kw.get("adam", False):
            self._update_layer = self.__update_layer_adam
        else:
            self._update_layer = self.__update_layer

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

    @property
    def last_gradient_norm(self: "MLP") -> float:
        """Gets the last computed gradient's norm."""
        if self._last_gradient is None:
            return None
        return np.linalg.norm(self._last_gradient)

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
        desc = {
            "cost": self._cost.activation,
            "learning_rate": self._lr,
            "preprocess": [{
                "parameters": self.encode_parameters(prepro[1]),
                "activation": prepro[0].__name__
            } for prepro in self._save_prepro],
            "layers": [{
                "dimension": layer.W.shape,
                "mono": layer.mono,
                "matrix": self.encode_matrix(layer.W),
                "activation": layer.activation
            } for layer in self._layers],
            "postprocess": [{
                "parameters": self.encode_parameters(postpro[1]),
                "activation": postpro[0].__name__
            } for postpro in self._save_postpro]
        }
        with open(path, 'w') as file:
            file.write(json.dumps(desc, indent=2))
        return self

    def update(self: "MLP", truth: ndarray, data: ndarray) -> Generator:
        """Perform one epoch of stochastic gradient descent.

        Updates weights using the current cost function and Adam optimizer.

        Args:
            truth (ndarray): Empirical target values (rows are samples).
            data (ndarray): Input samples (rows).
        Yields:
            ndarray: The output of the cost function after a forward pass.
        """
        truth = np.atleast_2d(truth)[:, None, :]
        data = np.atleast_2d(data)[:, None, :]
        gradient = None
        for (i, (y, x)) in enumerate(zip(truth, data)):
            print(f"\rPerforming iteration: {i}", end='')
            forward_pass = np.fromiter(self._forward_pass(x), ndarray)
            yield self._cost.eval(y, forward_pass[-1])
            if gradient is None:
                gradient = self._backpropagate(y, forward_pass)
            else:
                gradient += self._backpropagate(y, forward_pass)
            self._pb1 *= self._b1
            self._pb2 *= self._b2
        print()
        gradient /= data.shape[0]
        self._last_gradient = gradient

    def snap(self: "MLP") -> None:
        """Snapshot the MLP for revert."""
        for i, layer in enumerate(self._layers):
            self._last_matrices[i] = layer.W.copy()

    def revert(self: "MLP") -> None:
        """Reverts the mlp to the last computed matrices."""
        for i, layer in enumerate(self._layers):
            layer.W = self._last_matrices[i]

    def _backpropagate(self: "MLP", y: ndarray, input: ndarray) -> ndarray:
        """Backpropagate gradients and update internal moments.

        Args:
            y (ndarray): Truth values.
            input (ndarray): Sequence of inputs/outputs per layer, from input
                to final output.
        Returns:
            ndarray: The computed gradient.
        """
        dk = self._layers[-1].wdiff(input[-2])
        dk = dk @ self._cost.diff(y, input[-1]).T
        dk = np.atleast_1d(dk)
        self._update_layer(-1, np.outer(dk, input[-2]))
        for i in range(len(self._layers) - 2, -1, -1):
            dk = self._layers[i].wdiff(input[i]) @ self._layers[i + 1].W.T @ dk
            gradient = np.outer(dk, input[i])
            self._update_layer(i, gradient)
        return gradient

    def _forward_pass(self: "MLP", x: ndarray) -> Generator:
        """Yield inputs for each layer including the final output."""
        for layer in self._layers:
            yield x
            x = layer.eval(x)
        yield x

    def __update_layer(self: "MLP", i: int, gradient: ndarray) -> None:
        """Update a layer's weights using Adam.

        Args:
            i (int): Layer index.
            gradient (ndarray): Gradient for this layer.
        """
        self._layers[i].W -= self._lr * gradient

    def __update_layer_adam(self: "MLP", i: int, gradient: ndarray) -> None:
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

    @staticmethod
    def loadf(file: str) -> "MLP":
        """Load an MLP from a file.

        Args:
            file (str): Path to the file hodling the model description.
        Returns:
            MLP: Loaded instance.
        """
        with open(file, "r") as file:
            model = json.loads(file.read())
        return MLP.loadd(model)

    @staticmethod
    def loadd(model: dict) -> "MLP":
        """Load an MLP from a dict.

        Args:
            model (dict): Model description.
        Returns:
            MLP: Loaded instance.
        """
        arg = {
            "preprocess": [(
                getattr(Processor, x["activation"]),
                MLP.decode_parameters(x["parameters"])
            ) for x in model["preprocess"]],
            "layers": list(MLP._gen_layers(model["layers"])),
            "postprocess": [(
                getattr(Processor, x["activation"]),
                MLP.decode_parameters(x["parameters"])
            ) for x in model["postprocess"]],
            "cost": Neuron(model["cost"]),
            "learning_rate": model.get("learning_rate", 1e-3),
            "adam": model.get("adam", False)
        }
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
    def encode_matrix(mat: ndarray) -> str:
        """Encode a 2d matrix into a byte string.

        Args:
            mat (ndarray): The matrix to encode.
        Returns:
            bytes: The encoded bytestring.
        """
        packed = struct.pack('d' * mat.shape[0] * mat.shape[1], *mat.flatten())
        return base64.b64encode(packed).decode("ascii")

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
        unpacked = struct.unpack('d' * dim[0] * dim[1], mat)
        return np.array(unpacked).reshape(dim)

    @staticmethod
    def encode_parameters(prm: list) -> list:
        """Encode list of parameters for pre/post process functions.

        Args:
            prm (list): The list of parameter Returns:
            str: The encoded parameters as a string.
        """
        for i, p in enumerate(prm):
            match p:
                case list():
                    prm[i] = MLP.encode_parameters(p)
                case np.ndarray():
                    prm[i] = MLP.encode_parameters(p.tolist())
                case float():
                    prm[i] = 'd' + base64.b64encode(
                        struct.pack('d', p)
                    ).decode("ascii")
                case str():
                    prm[i] = 's' + str(len(p)) + 's' + base64.b64encode(
                        struct.pack(str(len(p)) + 's', p.encode("utf-8"))
                    ).decode("ascii")
        return list(prm)

    @staticmethod
    def decode_parameters(prm: list) -> list:
        """Decode list of parameters for pre/post process functions.

        Args:
            parameters (list): The encoded list.
        Returns:
            list (list): The decoded list.
        """
        for i, p in enumerate(prm):
            if isinstance(p, list):
                prm[i] = MLP.decode_parameters(p)
            elif isinstance(p, str):
                if p[0] == 'd':
                    prm[i] = struct.unpack('d', base64.b64decode(p[1:]))[0]
                elif p[0] == 's':
                    sep = p.split('s', maxsplit=2)
                    prm[i] = struct.unpack(
                        sep[1] + 's', base64.b64decode(sep[2])
                    )[0].decode("utf-8")
        return list(prm)

    @staticmethod
    def _gen_layers(descriptions: list) -> Generator:
        """Generate a list of Layer.

        Args:
            descriptions (list): List of layers descriptions.
        Yields:
            Layer: A single Layer object.
        """
        for desc in descriptions[:-1]:
            yield MLP._create_layer(desc)
        yield MLP._create_layer(descriptions[-1])

    @staticmethod
    def _create_layer(desc: dict) -> Layer:
        """Create a layer based on its description.

        Args:
            desc (dict): The description.
        Returns:
            Layer: The layer.
        """
        n, m = desc["dimension"]
        neuron = [Neuron(desc["activation"])]
        if desc["matrix"] is None:
            matrix = np.random.randn(n, m) * np.sqrt(2 / m)
        else:
            matrix = MLP.decode_matrix((n, m), desc["matrix"])
        if desc["mono"]:
            neurons = neuron
        else:
            neurons = neuron * (n - 1) + [Neuron("bias")]
        return Layer(neurons, matrix)


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
