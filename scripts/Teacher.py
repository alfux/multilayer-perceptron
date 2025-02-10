import argparse as arg
import sys
import traceback
from typing import Self

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from Layer import Layer
from MLP import MLP
from Neuron import Neuron
from Preprocessor import Preprocessor


class Teacher:
    """This class aims to find the best suited MLP to given parameters."""

    class BadTeacher(Exception):
        """Teacher specific exceptions."""
        pass

    def __init__(
            self: Self,
            book: DataFrame,
            target: str | int,
            normal: list = [0, 1],
            mlp: MLP = None
    ) -> None:
        """Creates an MLP Teacher.

        Args:
            <book> is a DataFrame containing the exercies and the lessons.
            <target> is the column containing the target values.
            <normal> is the normalization interval.
            <mlp> is a MLP instance of a MultilayerPerceptron.
        """
        self._prep = Preprocessor(book, target)
        self._prep.normalize(normal).add_bias()
        self._exercise: ndarray = self._prep.data
        self._lesson: ndarray = self._prep.target
        self._mlp: MLP = mlp

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
            <path> is the saving path of the trained MLP.
        """
        if self._mlp is None:
            raise Teacher.BadTeacher("No MLP loaded.")
        loss = Teacher.TCELF(self._lesson, self._mlp.eval(self._exercise))
        for i in range(epoch):
            print(f"Epoch {i}")
            self._mlp.update(self._lesson, self._exercise)
            loss = Teacher.TCELF(self._lesson, self._mlp.eval(self._exercise))
            print(f"loss = {loss}, LR = {self._mlp.learning_rate}")
            if loss < np.log(len(self._prep.unique)):
                break
        self._mlp.preprocess = self._prep.process
        self._mlp.save(path)
        return self

    def basic_regressor(self: Self) -> MLP:
        """Generates a basic MLP regressor based on the given DataFrame."""
        nx = self._exercise.shape[1]
        activation = Neuron(Teacher.ReLU, Teacher.dReLU)
        bias = Neuron(Teacher.bias, Teacher.dbias)
        matrix: ndarray = np.random.randn(nx, nx) * np.sqrt(2 / nx)
        matrix[-1] = np.zeros(matrix.shape[1])
        layers = [Layer([activation] * (nx - 1) + [bias], matrix)]
        for i in range(nx, 3, -1):
            matrix = np.random.randn(i - 1, i) * np.sqrt(2 / i)
            matrix[-1] = np.zeros(matrix.shape[1])
            layers += [Layer([activation] * (i - 2) + [bias], matrix)]
        matrix = np.random.randn(1, 3) * np.sqrt(2 / 3)
        layers += [Layer([activation], matrix)]
        mse = Teacher.MSE
        dmse = Teacher.dMSE
        return MLP(layers, Neuron(mse, dmse), learning_rate=1e-3)

    def basic_classifier(self: Self) -> MLP:
        """Generates a basic MLP classifier based on the given DataFrame."""
        (nx, ny) = (self._exercise.shape[1], len(self._prep.unique))
        activation = Neuron(Teacher.ReLU, Teacher.dReLU)
        bias = Neuron(Teacher.bias, Teacher.dbias)
        matrix: ndarray = np.random.randn(nx, nx) * np.sqrt(2 / nx)
        matrix[-1] = np.zeros(matrix.shape[1])
        layers = [Layer([activation] * (nx - 1) + [bias], matrix)]
        for i in range(nx, ny + 1, -1):
            matrix = np.random.randn(i - 1, i) * np.sqrt(2 / i)
            matrix[-1] = np.zeros(matrix.shape[1])
            layers += [Layer([activation] * (i - 2) + [bias], matrix)]
        activation = Neuron(Teacher.softmax, Teacher.dsoftmax)
        matrix = np.random.randn(ny, ny + 1) * np.sqrt(2 / (ny + 1))
        layers += [Layer([activation], matrix)]
        celf = Teacher.CELF
        dcelf = Teacher.dCELF
        return MLP(layers, Neuron(celf, dcelf), learning_rate=1e-3)

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
    def CELF(y: ndarray, x: ndarray) -> ndarray:
        """Cross Entropy Loss Function."""
        return -np.log(np.dot(y, x))

    @staticmethod
    def dCELF(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Cross Entropy Loss Function."""
        return -y / (x + 1e-15)

    @staticmethod
    def MSE(y: ndarray, x: ndarray) -> ndarray:
        """Mean Squared Error function."""
        return np.sum((y - x) ** 2) / np.size(x)

    @staticmethod
    def dMSE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Mean Squared Error function."""
        return -2 * (y - x) / np.size(x)

    @staticmethod
    def bias(x: ndarray) -> ndarray:
        """Returns 1."""
        return np.ones(np.atleast_1d(x).shape)

    @staticmethod
    def dbias(x: ndarray) -> ndarray:
        """Returns 0."""
        return np.zeros(np.atleast_1d(x).shape)


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
        av.add_argument("-n", default="[-1, 1]", help="normalization interval")
        av = av.parse_args()
        df = pd.read_csv(av.path, header=None if av.no_header else 0)
        df.columns = df.columns.map(str)
        df = df.drop(av.drops.split(';') if av.drops != '' else [], axis=1)
        teacher = Teacher(df, target=av.answer, normal=eval(av.n))
        teacher.mlp = teacher.basic_regressor()
        teacher.teach(epoch=1000)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
