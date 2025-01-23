import sys
from typing import Self
import argparse as arg

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray

from MLP import MLP


class Teacher:
    """This class aims to find the best suited MLP to given parameters."""

    def __init__(self: Self, book: DataFrame, **kwargs: dict) -> None:
        """Creates an MLP Teacher.

        Args:
            <book> is a DataFrame containing the lesson and the answers.
            <answer> is the column of the DataFrame containing the truth to
            train the MLP against.
            <mlp> is a MLP instance. Provide it as key word argument in order
            to train it instead of the basic generated MLP.
        """
        answer = kwargs["answer"] if "answer" in kwargs else book.columns[0]
        self._answer: ndarray = book[answer].to_numpy()
        self._lesson: ndarray = book.drop([answer], axis=1).to_numpy()
        self._mlp = kwargs["mlp"] if "mlp" in kwargs else self._gen_basic_mlp()

    def _gen_basic_mlp(self: Self) -> MLP:
        """Generates a basic MLP based on the given DataFrame."""
        n_in = self._lesson.shape[1]
        n_out = len(set(self._answer))
        mlp_string = f"Teacher.ReLU,Teacher.dReLU:{n_in}x{n_in}:M;"
        for i in range(n_in, n_out, -1):
            mlp_string += f"Teacher.ReLU,Teacher.dReLU:{i - 1}x{i}:M;"
        mlp_string += f"Teacher.softmax,Teacher.dsoftmax:{n_out}x{n_out}:S;"
        mlp_string += "Teacher.cross_entropy,Teacher.dcross_entropy;1e-3"
        print(mlp_string)

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
        return float("nan")

    @staticmethod
    def sigmoid(x: float) -> float:
        """Computes the value of sigmoid function in <x>."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: float) -> float:
        """Computes the differential of sigmoid in <x>."""
        return 1 / ((1 + np.exp(-x)) * (1 + np.exp(x)))

    @staticmethod
    def batch_cross_entropy(truth: ndarray, x: ndarray) -> float:
        """Batch cross entropy loss function."""
        return -np.sum(np.log(np.einsum("ij,ij->i", truth, x))) / x.shape[0]

    @staticmethod
    def dbatch_cross_entropy(truth: ndarray, x: ndarray) -> float:
        """Derivative of the cross entripy loss function."""
        return -np.sum(truth / (x + 1e-15)) / x.shape[0]

    @staticmethod
    def cross_entropy(truth: ndarray, x: ndarray) -> float:
        """Cross entropy loss function."""
        return -np.log(np.dot(truth, x))

    @staticmethod
    def dcross_entropy(truth: ndarray, x: ndarray) -> float:
        """Derivative of the cross entripy loss function."""
        return -truth / (x + 1e-15)


def main() -> int:
    """Docstring."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
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
        Teacher(df.drop(av.drops, axis=1), answer=av.answer)
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
