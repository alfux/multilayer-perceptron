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
        self._preprocessor = Preprocessor(book, **kwargs)
        self._lesson: ndarray = self._preprocessor.normalize().data
        self._answer: ndarray = self._preprocessor.to_onehot().onehot
        self._mlp: MLP = kwargs["mlp"] if "mlp" in kwargs else self._auto_mlp()
        self._mlp.preprocessor = self._preprocessor.process

    @property
    def mlp(self: Self) -> MLP:
        """Getter for the mlp model."""
        return self._mlp

    @mlp.setter
    def mlp(self: Self, value: MLP) -> None:
        """Setter for the mlp model."""
        self._mlp = value
        self._mlp.preprocessor = self._preprocessor._process

    def teach(self: Self, epoch: int = -1, e: float = 1e-3) -> Self:
        """Teaches the lesson to the internal MLP.

        Args:
            <epoch> is the number of epoch to realise with the training. When
            given <epoch> is 0 or less, precision is used to end the training.
            <e> is the precision under which the training stops because the
            Cross Entropy Loss is small enough.
        """
        if epoch <= 0:
            while Teacher.BCELF(self._answer, self._mlp(self._lesson)) < e:
                self._mlp.update(self._lesson)
        else:
            for _ in range(epoch):
                self._mlp.update(self._lesson)
        return self

    def _auto_mlp(self: Self) -> MLP:
        """Generates a basic MLP based on the given DataFrame."""
        n_in = self._lesson.shape[1]
        n_out = len(self._preprocessor.unique)
        neuron = Neuron(Teacher.ReLU, Teacher.dReLU)
        layers = [Layer([neuron] * n_in, rng.rand(n_in, n_in))]
        for i in range(n_in, n_out, -1):
            layers += [Layer([neuron] * (i - 1), rng.rand(i - 1, i))]
        neuron = Neuron(Teacher.softmax, Teacher.dsoftmax)
        layers += [Layer([neuron], rng.rand(n_out, n_out))]
        cel = Teacher.get_CELF(self._answer)
        dcel = Teacher.get_dCELF(self._answer)
        return MLP(layers, Neuron(cel, dcel))

    @staticmethod
    def softmax(x: ndarray) -> ndarray:
        """Computes the value of softmax function in <x>."""
        out = np.empty(len(x), float)
        for i in range(len(x)):
            out[i] = np.exp(x[i])
        norm = np.sum(out)
        return out / norm

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
    def ReLU(x: float) -> float:
        """Computes the value of Rectified Linear Unit function in <x>."""
        return np.max([0, x])

    @staticmethod
    def dReLU(x: float) -> float:
        """Computes the differential of ReLU in <x>."""
        if x != 0:
            return 1 if x > 0 else 0
        return 0

    @staticmethod
    def sigmoid(x: float) -> float:
        """Computes the value of sigmoid function in <x>."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: float) -> float:
        """Computes the differential of sigmoid in <x>."""
        return 1 / ((1 + np.exp(-x)) * (1 + np.exp(x)))

    @staticmethod
    def get_BCELF(y: ndarray) -> Callable[[ndarray], ndarray]:
        """Batched Cross Entropy Loss Function."""

        def BCELF(x: ndarray) -> ndarray:
            return -np.sum(np.log(np.einsum("ij,ij->i", y, x))) / x.shape[0]

        return BCELF

    @staticmethod
    def get_dBCELF(y: ndarray) -> Callable[[ndarray], ndarray]:
        """Derivative of the Batched Cross Entropy Loss Function."""

        def dBCELF(x: ndarray) -> ndarray:
            return -np.sum(y / (x + 1e-15)) / x.shape[0]

        return dBCELF

    @staticmethod
    def get_CELF(y: ndarray) -> Callable[[ndarray], ndarray]:
        """Cross Entropy Loss Function."""

        def CELF(x: ndarray) -> ndarray:
            return -np.log(np.dot(y, x))

        return CELF

    @staticmethod
    def get_dCELF(truth: ndarray) -> Callable[[ndarray], ndarray]:
        """Derivative of the Cross Entropy Loss Function."""

        def dCELF(x: ndarray) -> ndarray:
            return -truth / (x + 1e-15)

        return dCELF


def main() -> int:
    """Docstring."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("path", help="CSV file containing the lesson")
        help = "flag when the CSV file doesn't have headers"
        av.add_argument("--no-header", action="store_true", help=help)
        help = "answer column of the dataset"
        av.add_argument("answer", help=help)
        help = "semicolon separated list of drops"
        av.add_argument("drops", help=help, nargs='?', default='')
        av = av.parse_args()
        df = pd.read_csv(av.path, header=None if av.no_header else 0)
        av.drops = av.drops.split(';') if av.drops != '' else []
        df.columns = df.columns.map(str)
        teacher = Teacher(df.drop(av.drops, axis=1), answer=av.answer)
        teacher.teach(epoch=1)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
