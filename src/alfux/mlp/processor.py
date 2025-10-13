import sys
import traceback
from typing import Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from .statistics import Statistics


class Processor:
    """Preprocess datasets and targets for training.

    Provides utilities for normalization, standardization, one-hot encoding,
    and bias augmentation, while recording reversible transformations.
    """

    def __init__(
            self: "Processor", dataset: DataFrame, target: str | int
    ) -> None:
        """Create a preprocessor for a dataset.

        Args:
            dataset (DataFrame): Input dataset containing features and target.
            target (str | int): Name or index of the target column.
        """
        self._target: DataFrame = dataset.loc[:, [target]]
        self._data: DataFrame = dataset.drop([target], axis=1)
        constant = (self._data == self._data.iloc[0]).all(axis=0)
        self._data = self._data.loc[:, ~constant]
        self._stat: DataFrame = Statistics(self._data).stats
        self._unique: ndarray = None
        self._preprocess: list[Callable] = []
        self._postprocess: list[Callable] = []

    @property
    def target(self: "Processor") -> ndarray:
        """Get the target values as a NumPy array."""
        return self._target.to_numpy()

    @target.setter
    def target(self: "Processor", value: DataFrame | ndarray) -> None:
        """Set the current target values."""
        if isinstance(value, ndarray):
            self._target = DataFrame(value)
        else:
            self._target = value

    @property
    def data(self: "Processor") -> ndarray:
        """Get the current feature matrix as a NumPy array."""
        return self._data.to_numpy()

    @data.setter
    def data(self: "Processor", value: DataFrame | ndarray) -> None:
        """Set the current feature matrix and recompute stats."""
        if isinstance(value, ndarray):
            self._data = DataFrame(value)
        else:
            self._data = value
        self._stat = Statistics(self._data).stats

    @property
    def unique(self: "Processor") -> ndarray:
        """Get the ordered list of unique labels (if one-hot encoded)."""
        return self._unique

    @property
    def preprocess(self: "Processor") -> list[Callable]:
        """Get the list of performed preprocessing steps."""
        return self._preprocess

    @property
    def postprocess(self: "Processor") -> list[Callable]:
        """Get the list of performed postprocessing steps."""
        return self._postprocess

    def onehot(self: "Processor") -> "Processor":
        """One-hot encode the target labels and record the inverse mapping.

        Returns:
            Processor: The current instance.
        """
        if self._unique is None:
            (uni, inv) = np.unique(self._target, return_inverse=True)
            self._target = np.eye(len(uni))[inv.flatten()]
            self._target = DataFrame(self._target)
            self._unique = uni
            new_postprocess = [(Processor.revonehot, [uni])]
            self._postprocess = new_postprocess + self._postprocess
        return self

    def pre_standardize(self: "Processor") -> "Processor":
        """Standardize the dataset and record the process.

        Returns:
            Processor: The current instance.
        """
        mean = self._stat.loc["Mean"].to_numpy()
        std = self._stat.loc["Std"].to_numpy()
        self.data = Processor.standardize(mean, std, self._data)
        self._preprocess += [(Processor.standardize, [mean, std])]
        return self

    def post_standardize(self: "Processor") -> "Processor":
        """Standardize the target and record the inverse process.

        Returns:
            Processor: The current instance.
        """
        stats = Statistics(self._target).stats
        mean = stats.loc["Mean"].to_numpy()
        std = stats.loc["Std"].to_numpy()
        self.target = Processor.standardize(mean, std, self._target)
        new_postprocess = [(Processor.unstdardize, [mean, std])]
        self._postprocess = new_postprocess + self._postprocess
        return self

    def pre_normalize(
            self: "Processor", b: list[float] = [0, 1]
    ) -> "Processor":
        """Normalize the dataset to a given interval and record the process.

        Args:
            b (list[float], optional): Target interval ``[min, max]``.
                Defaults to ``[0, 1]``.

        Returns:
            Processor: The current instance.
        """
        m = self._stat.loc["Min"].to_numpy()
        s = self._stat.loc["Max"].to_numpy() - m
        self.data = Processor.normalize(m, s, b, self._data)
        self._preprocess += [(Processor.normalize, [m, s, b])]
        return self

    def post_normalize(
            self: "Processor", b: list[float] = [0, 1]
    ) -> "Processor":
        """Normalize the target and record the inverse process.

        Returns:
            Processor: The current instance.
        """
        stats = Statistics(self._target).stats
        m = stats.loc["Min"].to_numpy()
        s = stats.loc["Max"].to_numpy() - m
        self.target = Processor.normalize(m, s, b, self._target)
        new_postprocess = [(Processor.unrmalize, [m, s, b])]
        self._postprocess = new_postprocess + self._postprocess
        return self

    def pre_bias(self: "Processor") -> "Processor":
        """Adds a bias component at the end of the vector of the dataset."""
        self.data = Processor.add_bias(self._data)
        self._preprocess += [(Processor.add_bias, [])]
        return self

    @staticmethod
    def compile_processes(processes: list[Callable]) -> Callable:
        """Compile a list of processes into a single function.

        Args:
            processes (list[tuple[Callable, list]]): Sequence of
                ``(function, [params...])`` pairs applied in order.

        Returns:
            Callable: Composed processing function.
        """
        if len(processes) == 0:
            return Processor.identity

        def result(x: ndarray) -> ndarray:
            """Apply the first process."""
            return processes[0][0](*processes[0][1], x)

        for process in processes[1:]:
            prev = result

            def result(x: ndarray) -> ndarray:
                """Apply the next process on previous output."""
                return process[0](*process[1], prev(x))

        return result

    @staticmethod
    def standardize(mean: ndarray, std: ndarray, x: ndarray) -> ndarray:
        """Standardize an array.

        Args:
            mean (ndarray): Mean of the data to standardize.
            std (ndarray): Standard deviation of the data.
            x (ndarray): Input data.

        Returns:
            ndarray: Standardized output ``(x - mean) / std``.
        """
        return (x - mean) / std

    @staticmethod
    def unstdardize(mean: ndarray, std: ndarray, x: ndarray) -> ndarray:
        """Revert standardization.

        Args:
            mean (ndarray): Original mean.
            std (ndarray): Original standard deviation.
            x (ndarray): Standardized data.

        Returns:
            ndarray: Unstandardized output ``std * x + mean``.
        """
        return std * x + mean

    @staticmethod
    def normalize(min: ndarray, span: ndarray, b: list, x: ndarray) -> ndarray:
        """Normalize an array to a target interval.

        Args:
            min (ndarray): Minimum of the original data.
            span (ndarray): Range ``max - min`` of the original data.
            b (list): Target interval ``[min, max]``.
            x (ndarray): Input data.

        Returns:
            ndarray: Normalized output.
        """
        return b[0] + (b[1] - b[0]) * (x - min) / span

    @staticmethod
    def unrmalize(min: ndarray, span: ndarray, b: list, x: ndarray) -> ndarray:
        """Revert normalization.

        Args:
            min (ndarray): Original minimum.
            span (ndarray): Original range ``max - min``.
            b (list): Target interval ``[min, max]`` used for normalization.
            x (ndarray): Normalized data.

        Returns:
            ndarray: Unnormalized output.
        """
        return min + span * (x - b[0]) / (b[1] - b[0])

    @staticmethod
    def revonehot(uniques: ndarray, x: ndarray) -> ndarray:
        """Reverse one-hot encoding to original labels.

        Args:
            uniques (ndarray): Ordered list of unique labels.
            x (ndarray): One-hot encoded values.

        Returns:
            ndarray: Decoded labels corresponding to each row of ``x``.
        """
        print(x)
        return uniques[np.argmax(x, axis=1)]

    @staticmethod
    def add_bias(x: ndarray) -> ndarray:
        """Append a bias column of ones to an array."""
        if x.ndim == 2:
            return np.column_stack([x, np.ones((x.shape[0], 1))])
        return np.concat([x, np.ones(1)])

    @staticmethod
    def identity(x: ndarray) -> ndarray:
        """Return the input unchanged."""
        return x

    @staticmethod
    def map(labels: str, x: ndarray) -> str:
        """Map the highest value of an ndarray to a string.

        Args:
            labels (str): Labels for the mapping.
            x (ndarray): A list of value (distribution).
        Returns:
            str: The corresponding string.
        """
        return labels[np.argmax(x)]


def main() -> int:
    """Print a demonstration of Processor functionalities.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
    try:
        book = DataFrame([['a', 1, 0, 0.8],
                          ['b', 4, 375, 7],
                          ['a', 9, -3, 12]])
        processor = Processor(book)
        print("Data", book, sep="\n\n", end="\n\n")
        processor.pre_standardize()
        print("Standaridzed", processor.data, sep="\n\n", end="\n\n")
        processor.pre_normalize([-1, 1])
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
