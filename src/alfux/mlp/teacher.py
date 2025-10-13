import argparse as arg
from datetime import datetime
import sys
import traceback

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame

from .mlp import MLP, Layer, Neuron
from .processor import Processor


class Teacher:
    """Train an MLP using provided data and configuration."""

    class BadTeacher(Exception):
        """Teacher-specific exception type."""
        pass

    def __init__(
            self: "Teacher", book: DataFrame, target: str | int,
            normal: list = [0, 1], mlp: MLP = None, pre: str = "normalize",
            post: str = "normalize", bias: bool = True
    ) -> None:
        """Create a Teacher for an MLP.

        Args:
            book (DataFrame): DataFrame containing features and target.
            target (str | int): Column containing target values.
            normal (list, optional): Normalization interval. Defaults to
                ``[0, 1]``.
            mlp (MLP | None, optional): MLP to train. Defaults to ``None``.
            pre (str, optional): Preprocess type: ``"normalize"`` or
                ``"standardize"``. Defaults to ``"normalize"``.
            post (str, optional): Postprocess type: ``"normalize"``,
                ``"standardize"`` or ``"onehot"``. Defaults to
                ``"normalize"``.
            bias (bool, optional): Whether to add a bias feature. Defaults to
                ``True``.
        """
        self._proc = Processor(book, target)
        if pre == "normalize":
            self._proc.pre_normalize(normal)
        elif pre == "standardize":
            self._proc.pre_standardize()
        if post == "normalize":
            self._proc.post_normalize(normal)
        elif post == "standardize":
            self._proc.post_standardize()
        elif post == "onehot":
            self._proc.onehot()
        if bias:
            self._proc.pre_bias()
        self._data: ndarray = self._proc.data
        self._truth: ndarray = self._proc.target
        self._mlp: MLP = mlp

    @property
    def mlp(self: "Teacher") -> MLP:
        """Get the underlying MLP model."""
        return self._mlp

    @mlp.setter
    def mlp(self: "Teacher", value: MLP) -> None:
        """Set the underlying MLP model."""
        self._mlp = value

    def teach(
        self: "Teacher", epoch: int, time: bool = False, frac: float = 1
    ) -> "Teacher":
        """Train the internal MLP for a number of epochs.

        Args:
            epoch (int): Number of training epochs.
            time (bool, optional): Print training duration. Defaults to
                ``False``.
            frac (float, optional): Sample fraction per epoch.
                Defaults to ``1``.

        Returns:
            Teacher: The current instance.
        """
        if self._mlp is None:
            raise Teacher.BadTeacher("No MLP loaded.")
        t = datetime.now()
        for i in range(epoch):
            print(f"\nEpoch {i}:")
            self._mlp.update(*self._sample(frac))
        self._mlp.preprocess = self._proc.preprocess
        self._mlp.postprocess = self._proc.postprocess
        if time:
            print("\n\tTraining time:", datetime.now() - t)
        return self

    def _sample(self: "Teacher", frac: float) -> tuple[ndarray, ndarray]:
        """Select a random sample of the data.

        Args:
            frac (float): Proportion of the dataset to sample.

        Returns:
            tuple[ndarray, ndarray]: ``(truth, data)`` mini-batch.
        """
        size = np.int64(np.round(np.clip(frac, 0, 1) * self._data.shape[0]))
        idx = np.random.choice(self._data.shape[0], size=size, replace=False)
        return (self._truth[idx], self._data[idx])


def basic_regressor(nx: int) -> MLP:
    """Generate a basic MLP regressor with decreasing layer sizes.

    Args:
        nx (int): Number of inputs including bias.

    Returns:
        MLP: Configured MLP instance.
    """
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


def main() -> int:
    """Train a simple regression MLP on a CSV.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
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
        teacher.mlp = basic_regressor(df.shape[1])
        teacher.teach(epoch=1).mlp.save("regression_conso.mlp")
        teacher.mlp = MLP.load("regression_conso.mlp")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
