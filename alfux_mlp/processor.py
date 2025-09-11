import sys
import traceback
from typing import Self, Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from .statistics import Statistics


class Processor:
    """Create an instance of this class to preprocess a dataset."""

    def __init__(self: Self, dataset: DataFrame, target: str | int) -> None:
        """Creates a preprocessor for a given dataset.

        Args:
            dataset is the DataFrame to process.
            target is the column containing the learning target.
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
    def target(self: Self) -> ndarray:
        """Getter for the target feature."""
        return self._target.to_numpy()

    @target.setter
    def target(self: Self, value: DataFrame | ndarray) -> None:
        """Setter of the current target."""
        if isinstance(value, ndarray):
            self._target = DataFrame(value)
        else:
            self._target = value

    @property
    def data(self: Self) -> ndarray:
        """Getter of the current dataset."""
        return self._data.to_numpy()

    @data.setter
    def data(self: Self, value: DataFrame | ndarray) -> None:
        """Setter of the current dataset."""
        if isinstance(value, ndarray):
            self._data = DataFrame(value)
        else:
            self._data = value
        self._stat = Statistics(self._data).stats

    @property
    def unique(self: Self) -> ndarray:
        """Getter of the ordered list of unique labels."""
        return self._unique

    @property
    def preprocess(self: Self) -> list[Callable]:
        """Getter of the ordered list of performed preprocesses"""
        return self._preprocess

    @property
    def postprocess(self: Self) -> list[Callable]:
        """Getter of the ordered list of performed postprocesses"""
        return self._postprocess

    def onehot(self: Self) -> Self:
        """Transform the label field in a vectorized equivalent."""
        if self._unique is None:
            (uni, inv) = np.unique(self._target, return_inverse=True)
            self._target = np.eye(len(uni))[inv.flatten()]
            self._target = DataFrame(self._target)
            self._unique = uni
        return self

    def pre_standardize(self: Self) -> Self:
        """Standardizes the current dataset and stores the process.

        <b>Returns:</b>
            The current instance of the class.
        """
        mean = self._stat.loc["Mean"].to_numpy()
        std = self._stat.loc["Std"].to_numpy()
        self.data = Processor.standardize(mean, std, self._data)
        self._preprocess += [(Processor.standardize, [mean, std])]
        return self

    def post_standardize(self: Self) -> Self:
        """Standardizes the current target and stores the inverse process.

        <b>Returns:</b>
            The current instance of the class.
        """
        stats = Statistics(self._target).stats
        mean = stats.loc["Mean"].to_numpy()
        std = stats.loc["Std"].to_numpy()
        self.target = Processor.standardize(mean, std, self._target)
        new_postprocess = [(Processor.unstdardize, [mean, std])]
        self._postprocess = new_postprocess + self._postprocess
        return self

    def pre_normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current dataset and stores the process.

        Args:
            <b> represents the wanted interval.
        Returns:
            The current instance of the class.
        """
        m = self._stat.loc["Min"].to_numpy()
        s = self._stat.loc["Max"].to_numpy() - m
        self.data = Processor.normalize(m, s, b, self._data)
        self._preprocess += [(Processor.normalize, [m, s, b])]
        return self

    def post_normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current target and stores the inverse process.

        <b>Returns:</b>
            The current instance of the class.
        """
        stats = Statistics(self._target).stats
        m = stats.loc["Min"].to_numpy()
        s = stats.loc["Max"].to_numpy() - m
        self.target = Processor.normalize(m, s, b, self._target)
        new_postprocess = [(Processor.unrmalize, [m, s, b])]
        self._postprocess = new_postprocess + self._postprocess
        return self

    def pre_bias(self: Self) -> Self:
        """Adds a bias component at the end of the vector of the dataset."""
        self.data = Processor.add_bias(self._data)
        self._preprocess += [(Processor.add_bias, [])]
        return self

    @staticmethod
    def compile_processes(processes: list[Callable]) -> Callable:
        """Compiles a list of processes in a single function.

        Args:
            processes is the list of tuples containing a function and its
            parameters.
        """
        if len(processes) == 0:
            return Processor.identity

        def result(x: ndarray) -> ndarray:
            """First process"""
            return processes[0][0](*processes[0][1], x)

        for process in processes[1:]:
            prev = result

            def result(x: ndarray) -> ndarray:
                """Composed process"""
                return process[0](*process[1], prev(x))

        return result

    @staticmethod
    def standardize(mean: ndarray, std: ndarray, x: ndarray) -> ndarray:
        """Standardizes a ndarray.

        Args:
            mean is the mean of the datas to standardize
            std is the standard deviation of the datas to standardize
            x is the input to standardize
        Returns:
            The standardized output
        """
        return (x - mean) / std

    @staticmethod
    def unstdardize(mean: ndarray, std: ndarray, x: ndarray) -> ndarray:
        """Revert the standardization of a ndarray.

        Args:
            mean is the mean of the datas before standardization
            std is the standard deviation of the datas before standardization
            x is the input to unstandardize
        Returns:
            The unstandardized output
        """
        return std * x + mean

    @staticmethod
    def normalize(min: ndarray, span: ndarray, b: list, x: ndarray) -> ndarray:
        """Normalizes a ndarray.

        Args:
            min is the min of the datas to normalize
            span is the Max - Min range of the datas to normalize
            x is the input to normalize
        Returns:
            The normalized output
        """
        return b[0] + (b[1] - b[0]) * (x - min) / span

    @staticmethod
    def unrmalize(min: ndarray, span: ndarray, b: list, x: ndarray) -> ndarray:
        """Revert the normalization of a ndarray.

        Args:
            min is the min of the datas before normalization
            span is the Max - Min range of the datas before normalization
            x is the input to unnormalize
        Returns:
            The unnormalized output
        """
        return min + span * (x - b[0]) / (b[1] - b[0])

    @staticmethod
    def add_bias(x: ndarray) -> ndarray:
        """Adds a column of 1 at the right end of a ndarray."""
        if x.ndim == 2:
            return np.column_stack([x, np.ones((x.shape[0], 1))])
        return np.concat([x, np.ones(1)])

    @staticmethod
    def identity(x: ndarray) -> ndarray:
        """Returns the input."""
        return x


def main() -> int:
    """Print a test of the Processor functionalities"""
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
