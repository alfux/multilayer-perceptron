import argparse as arg
import sys
import traceback
from typing import Self, Callable

import numpy as np
from numpy import ndarray
import numpy.random as rng
import pandas as pd
from pandas import DataFrame

from Layer import Layer
from MLP import MLP
from Neuron import Neuron
from Preprocessor import Preprocessor


class Teacher:
    """This class aims to find the best suited MLP to given parameters."""

    def __init__(self: Self, book: DataFrame, **kwargs: dict) -> None:
        """Creates an MLP Teacher.

        Args:
            <book> is a DataFrame containing the lesson and the answers.
        Kwargs:
            <mlp> is a MLP instance. Provide it as key word argument in order
            to train it instead of the basic generated MLP.
        """
        self._prep = Preprocessor(book, **kwargs)
        if "normal" in kwargs:
            self._lesson: ndarray = self._prep.normalize(kwargs["normal"]).data
        else:
            self._lesson: ndarray = self._prep.data
        self._truth: ndarray = self._prep.to_onehot().onehot
        self._mlp: MLP = kwargs["mlp"] if "mlp" in kwargs else self._auto_mlp()

    @property
    def mlp(self: Self) -> MLP:
        """Getter for the mlp model."""
        return self._mlp

    @mlp.setter
    def mlp(self: Self, value: MLP) -> None:
        """Setter for the mlp model."""
        self._mlp = value

    def teach(self: Self, epoch: int, path: str = "./unnamed.mlp") -> Self:
        """Teaches the lesson to the internal MLP.

        Args:
            <epoch> is the number of epoch to realise with the training. When
            given <epoch> is 0 or less, precision is used to end the training.
            <e> is the precision under which the training stops because the
            Cross Entropy Loss is small enough.
            <path> is the saving path of the trained MLP.
        """

        loss = Teacher.TCELF(self._truth, self._mlp.eval(self._lesson))
        for _ in range(epoch):
            self._mlp.update(self._truth, self._lesson)
            loss = Teacher.TCELF(self._truth, self._mlp.eval(self._lesson))
            print(f"loss = {loss}, LR = {self._mlp.learning_rate}")
        print(Teacher.TCELF(self._truth, self._mlp.eval(self._lesson)))
        self._mlp.preprocess = self._prep.process
        self._mlp.save(path)
        return self

    def _auto_mlp(self: Self) -> MLP:
        """Generates a basic MLP based on the given DataFrame."""
        (nx, ny) = (self._lesson.shape[1], len(self._prep.unique))
        neuron = Neuron(Teacher.ReLU, Teacher.dReLU)
        layers = [Layer([neuron] * nx, rng.randn(nx, nx + 1) * np.sqrt(2 / (nx + 1)))]
        for i in range(nx, ny, -1):
            layers += [Layer(
                [neuron] * (i - 1),
                rng.randn(i - 1, i + 1) * np.sqrt(2 / (i + 1))
            )]
        neuron = Neuron(Teacher.softmax, Teacher.dsoftmax)
        layers += [Layer([neuron], rng.randn(ny, ny + 1) * np.sqrt(2 / (ny + 1)))]
        cel = Teacher.CELF
        dcel = Teacher.dCELF
        return MLP(layers, Neuron(cel, dcel), learning_rate=1e-4)

    @staticmethod
    def softmax(x: ndarray) -> ndarray:
        """Computes the value of softmax function in <x>."""
        out = np.exp(x)
        if x.ndim > 1:
            return out / np.atleast_2d(np.sum(out, axis=1)).T
        return out / np.sum(out)

    @staticmethod
    def dsoftmax(x: ndarray) -> ndarray:
        """Compute the differential of softmax in <x>."""
        vector = np.exp(x)
        vector /= np.sum(vector)
        Jacobian = -np.outer(vector, vector)
        index = np.arange(len(x))
        Jacobian[index, index] += vector
        return Jacobian

    @staticmethod
    def ReLU(x: ndarray) -> ndarray:
        """Computes the value of Rectified Linear Unit function in <x>."""
        return (x > 0).astype(float) * x

    @staticmethod
    def dReLU(x: ndarray) -> ndarray:
        """Computes the differential of ReLU in <x>."""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: float) -> float:
        """Computes the value of sigmoid function in <x>."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: float) -> float:
        """Computes the differential of sigmoid in <x>."""
        return 1 / ((1 + np.exp(-x)) * (1 + np.exp(x)))

    @staticmethod
    def TCELF(y: ndarray, x: ndarray) -> ndarray:
        """Total Cross Entropy Loss Function."""
        return -np.sum(np.log(np.einsum("ij,ij->i", y, x))) / x.shape[0]

    @staticmethod
    def dTCELF(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Total Cross Entropy Loss Function."""
        return -np.sum(y / (x + 1e-15)) / x.shape[0]

    @staticmethod
    def CELF(y: ndarray, x: ndarray) -> Callable[[ndarray], ndarray]:
        """Cross Entropy Loss Function."""
        return -np.log(np.dot(y, x))

    @staticmethod
    def dCELF(y: ndarray, x: ndarray) -> Callable[[ndarray], ndarray]:
        """Derivative of the Cross Entropy Loss Function."""
        return -y / (x + 1e-15)


def main() -> int:
    """Docstring."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("path", help="CSV file containing the lesson")
        help = "flag when the CSV file doesn't have headers"
        av.add_argument("--no-header", action="store_true", help=help)
        av.add_argument("answer", help="answer column of the dataset")
        help = "semicolon separated list of drops"
        av.add_argument("drops", help=help, nargs='?', default='')
        av.add_argument("-n", default="[-1, 1]", help="normal parameter")
        av = av.parse_args()
        df = pd.read_csv(av.path, header=None if av.no_header else 0)
        df.columns = df.columns.map(str)
        df = df.drop(av.drops.split(';') if av.drops != '' else [], axis=1)
        Teacher(df, answer=av.answer, normal=eval(av.n)).teach(epoch=1000)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
