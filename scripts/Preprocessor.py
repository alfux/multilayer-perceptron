import sys
import traceback
from typing import Self, Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from Statistics import Statistics


class Preprocessor:
    """Create an instance of this class to preprocess a dataset."""

    def __init__(self: Self, dataset: DataFrame, **kwargs: dict) -> None:
        """Creates a preprocessor for a given dataset.

        Args:
            <dataset> is the DataFrame to preprocess.
        Kwargs:
            <labels> is the column of the DataFrame containing classification
            characters.
        """
        labels = kwargs["labels"] if "labels" in kwargs else dataset.columns[0]
        self._onehot: DataFrame = dataset[labels]
        self._unique: ndarray = None
        self._data: DataFrame = dataset.drop([labels], axis=1)
        self._stat: DataFrame = Statistics(self._data).stats
        self._process: Callable = Preprocessor.identity

    @property
    def onehot(self: Self) -> ndarray:
        """Getter for the onehot encoded labels."""
        return self._onehot

    @property
    def unique(self: Self) -> ndarray:
        """Getter the ordered list of unique labels."""
        return self._unique

    @property
    def data(self: Self) -> ndarray:
        """Getter of the current unprocessed dataset."""
        return self._data.to_numpy()

    @data.setter
    def data(self: Self, value) -> None:
        """Setter of the current dataset."""
        self._data = value
        self._stat = Statistics(self._data).stats

    @property
    def process(self: Self) -> Callable[[ndarray], ndarray]:
        """Returns the composition of process applied over the dataset."""
        return self._process

    def to_onehot(self: Self) -> Self:
        """Transform the label field in a vectorized equivalent."""
        if self._unique is None:
            (uni, inv) = np.unique(self._onehot, return_inverse=True)
            self._onehot = np.eye(len(uni))[inv]
            self._unique = uni
        return self

    def standardize(self: Self) -> Self:
        """Standardizes the current dataset and stores the process."""
        mean = self._stat.loc["Mean"].to_numpy()
        std = self._stat.loc["Std"].to_numpy()
        return self._apply(lambda x: (x - mean) / std)

    def normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current dataset and stores the process.

        Args:
            <b> represents the wanted interval.
        Returns:
            The current instance of the class.
        """
        min = self._stat.loc["Min"].to_numpy()
        scale = self._stat.loc["Max"].to_numpy() - min
        return self._apply(lambda x: b[0] + (b[1] - b[0]) * (x - min) / scale)

    def _apply(self: Self, func: Callable) -> Self:
        """Apply given <func> to <self._data>."""
        if self._process == Preprocessor.identity:
            self._process = func
        else:
            previous_process = self._process
            self._process = lambda x: func(previous_process(x))
        self.data = func(self._data)
        return self

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
        print(processor.onehot, end="\n\n")
        processor.to_onehot()
        print(processor.onehot, end="\n\n")
        return 0
    except Exception as err:
        print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
