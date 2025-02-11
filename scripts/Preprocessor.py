import sys
import traceback
from typing import Self, Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from Statistics import Statistics


class Preprocessor:
    """Create an instance of this class to preprocess a dataset."""

    def __init__(self: Self, dataset: DataFrame, target: str | int) -> None:
        """Creates a preprocessor for a given dataset.

        Args:
            <dataset> is the DataFrame to preprocess.
            <labels> is the column of the DataFrame containing classification
            characters.
        """
        if target is None:
            target = dataset.columns[0]
        self._target: DataFrame = dataset[target].to_numpy()
        self._unique: ndarray = None
        self._data: DataFrame = dataset.drop([target], axis=1)
        constant = (self._data == self._data.iloc[0]).all(axis=0)
        self._data = self._data.loc[:, ~constant]
        self._stat: DataFrame = Statistics(self._data).stats
        self._process: Callable = Preprocessor.identity
        self._repr = "lambda x: x"

    def __call__(self: Self, x: ndarray) -> ndarray:
        "Calling method of the preprocessor."
        return self._process(x)

    def __repr__(self: Self) -> str:
        """String representation of the object. Can be used with eval()."""
        return self._repr

    @property
    def target(self: Self) -> ndarray:
        """Getter for the target feature."""
        return self._target

    @property
    def unique(self: Self) -> ndarray:
        """Getter of the ordered list of unique labels."""
        return self._unique

    @property
    def data(self: Self) -> ndarray:
        """Getter of the current unprocessed dataset."""
        if isinstance(self._data, ndarray):
            return self._data
        return self._data.to_numpy()

    @data.setter
    def data(self: Self, value) -> None:
        """Setter of the current dataset."""
        self._data = value
        if isinstance(self._data, ndarray):
            self._stat = Statistics(DataFrame(self._data)).stats
        else:
            self._stat = Statistics(self._data).stats

    def to_onehot(self: Self) -> Self:
        """Transform the label field in a vectorized equivalent."""
        if self._unique is None:
            (uni, inv) = np.unique(self._target, return_inverse=True)
            self._target = np.eye(len(uni))[inv]
            self._unique = uni
        return self

    def standardize(self: Self) -> Self:
        """Standardizes the current dataset and stores the process.

        Returns:
            The current instance of the class.
        """
        mean = self._stat.loc["Mean"].to_numpy()
        std = self._stat.loc["Std"].to_numpy()
        return self._apply(f"lambda x: (x - {mean}) / {std})")

    def normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current dataset and stores the process.

        Args:
            <b> represents the wanted interval.
        Returns:
            The current instance of the class.
        """
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        minium = self._stat.loc["Min"].to_numpy()
        scale = self._stat.loc["Max"].to_numpy() - minium
        strepr = f"lambda x: {b[0]} + {b[1] - b[0]} * (x - np."
        strepr += f"{repr(minium)}) / np.{repr(scale)}"
        np.set_printoptions()
        return self._apply(strepr)

    def add_bias(self: Self) -> Self:
        """Adds a bias component at the end of the vector."""
        return self._apply("Preprocessor.adding_bias")

    def _apply(self: Self, strepr: str) -> Self:
        """Apply given function to the data.

        Args:
            <func> function to apply.
            <strepr> is a string representation of the added preprocess.
        """
        func = eval(strepr)
        if self._process == Preprocessor.identity:
            self._process = func
            self._repr = strepr
        else:
            previous_process = self._process
            self._process = lambda x: func(previous_process(x))
            self._repr = self._str_compose(strepr)
        self.data = func(self._data)
        return self

    def _str_compose(self: Self, strepr: str) -> str:
        """Compose the new str representation with the old one.
        Args:
            The new added process as a string.
        Returns:
            The composition as a string.
        """
        if ':' in strepr:
            (prototype, body) = strepr.split(':')
            (prefix, suffix) = body.split('x')
            return prototype + prefix + f"({self._repr})(x)" + suffix
        else:
            return f"lambda x: {strepr}(({self._repr})(x))"

    @staticmethod
    def adding_bias(x: ndarray) -> ndarray:
        if x.ndim == 2:
            return np.column_stack([x, np.ones((x.shape[0], 1))])
        return np.concat([x, np.ones(1)])

    @staticmethod
    def identity(x: ndarray) -> ndarray:
        return x


def main() -> int:
    """Print a test of the Preprocessor functionalities"""
    try:
        book = DataFrame([['a', 1, 0, 0.8],
                          ['b', 4, 375, 7],
                          ['a', 9, -3, 12]])
        processor = Preprocessor(book)
        print("Data", book, sep="\n\n", end="\n\n")
        processor.standardize()
        print("Standaridzed", processor.data, sep="\n\n", end="\n\n")
        processor.normalize([-1, 1])
        print("Normalized [-1, 1]", processor.data, sep="\n\n", end="\n\n")
        data = book.drop([0], axis=1)
        print("Reset", book, sep="\n\n", end="\n\n")
        print("Composition", processor.process(data), sep="\n\n", end="\n\n")
        print(processor.target, end="\n\n")
        processor.to_onehot()
        print(processor.target, end="\n\n")
        return 0
    except Exception as err:
        print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
