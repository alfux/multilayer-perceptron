import argparse as arg
import sys
import traceback
from typing import Self

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from MLP import MLP, Layer, Neuron
from Processor import Processor


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
        """<b>Creates an MLP Teacher</b>.

        <b>Args:</b>
            <b>book</b> is a DataFrame containing the datas and target.
            <b>target</b> is the column containing the target values.
            <b>normal</b> is the normalization interval.
            <b>mlp</b> is a MLP instance of a MultilayerPerceptron.
        <b>Returns:</b>
            Nothing
        """
        self._prep = Processor(book, target)
        self._prep.pre_normalize(normal).post_normalize(normal).add_bias()
        self._data: ndarray = self._prep.data
        self._true: ndarray = self._prep.target
        self._mlp: MLP = mlp

    @property
    def mlp(self: Self) -> MLP:
        """Getter for the mlp model."""
        return self._mlp

    @mlp.setter
    def mlp(self: Self, value: MLP) -> None:
        """Setter for the mlp model."""
        self._mlp = value

    def teach(self: Self, epoch: int) -> Self:
        """Teaches the lesson to the internal MLP.

        Args:
            <epoch> is the number of epoch to realise with the training. When
            given <epoch> is 0 or less, precision is used to end the training.
        """
        if self._mlp is None:
            raise Teacher.BadTeacher("No MLP loaded.")
        loss = self._mlp.cost.eval(self._true, self._mlp.eval(self._data))
        print(f"Initial loss = {loss}, LR = {self._mlp.learning_rate}\n")
        for i in range(epoch):
            print(f"\nEpoch {i}:")
            self._mlp.update(self._true, self._data)
            output = self._mlp.eval(self._data)
            loss = self._mlp.cost.eval(self._true, output)
            grad = np.linalg.norm(self._mlp.cost.diff(self._true, output))
            print(f"\nLoss = {loss} - Grad = {grad}")
            if grad < 1e-3:
                break
        self._mlp.preprocess = self._prep.prestr
        self._mlp.postprocess = self._prep.poststr
        return self

    def save(self: "Teacher", path: str = "default.mlp") -> "Teacher":
        """Saves the current mlp into the file in <b>path</b>."""
        self._mlp.preprocess = self._prep.prestr
        self._mlp.postprocess = self._prep.poststr
        self._mlp.save(path)
        return self

    def basic_regressor(self: Self) -> MLP:
        """Generates a basic MLP regressor based on the given DataFrame."""
        nx = self._data.shape[1]
        activation = Neuron("Neuron.ReLU", "Neuron.dReLU")
        bias = Neuron("Neuron.bias", "Neuron.dbias")
        matrix: ndarray = np.random.randn(nx, nx) * np.sqrt(2 / nx)
        matrix[-1] = np.zeros(matrix.shape[1])
        layers = [Layer([activation] * (nx - 1) + [bias], matrix)]
        for i in range(nx, 3, -1):
            matrix = np.random.randn(i - 1, i) * np.sqrt(2 / i)
            matrix[-1] = np.zeros(matrix.shape[1])
            layers += [Layer([activation] * (i - 2) + [bias], matrix)]
        matrix = np.random.randn(1, 3) * np.sqrt(2 / 3)
        layers += [Layer([activation], matrix)]
        cost = Neuron("Neuron.MSE", "Neuron.dMSE")
        return MLP(layers, cost, learning_rate=1e-3)

    def basic_classifier(self: Self) -> MLP:
        """Generates a basic MLP classifier based on the given DataFrame."""
        (nx, ny) = (self._data.shape[1], len(self._prep.unique))
        activation = Neuron("Neuron.ReLU", "Neuron.dReLU")
        bias = Neuron("Neuron.bias", "Neuron.dbias")
        matrix: ndarray = np.random.randn(nx, nx) * np.sqrt(2 / nx)
        matrix[-1] = np.zeros(matrix.shape[1])
        layers = [Layer([activation] * (nx - 1) + [bias], matrix)]
        for i in range(nx, ny + 1, -1):
            matrix = np.random.randn(i - 1, i) * np.sqrt(2 / i)
            matrix[-1] = np.zeros(matrix.shape[1])
            layers += [Layer([activation] * (i - 2) + [bias], matrix)]
        activation = Neuron("Neuron.softmax", "Neuron.dsoftmax")
        matrix = np.random.randn(ny, ny + 1) * np.sqrt(2 / (ny + 1))
        layers += [Layer([activation], matrix)]
        cost = Neuron("Neuron.CELF", "Neuron.dCELF")
        return MLP(layers, cost, learning_rate=1e-3)


def main() -> int:
    """Trains an MLP."""
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
        teacher.teach(epoch=1).save("regression_conso.mlp")
        teacher.mlp = MLP.load("regression_conso.mlp")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
