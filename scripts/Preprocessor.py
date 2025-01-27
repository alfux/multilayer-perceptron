import sys
from typing import Self, Callable

from numpy import ndarray
from pandas import DataFrame

from Statistics import Statistics


class Preprocessor:
    """Create an instance of this class to preprocess a dataset."""

    def __init__(self: Self, dataset: DataFrame) -> None:
        """Creates a preprocessor for a given dataset.

        Args:
            <dataset> is the DataFrame to preprocess.
        """
        self._data = dataset
        self._stat = Statistics(dataset).stats
        self._process = Preprocessor.identity

    @property
    def data(self: Self) -> ndarray:
        """Getter of the current unprocessed dataset."""
        return self._data.to_numpy()

    @property
    def process(self: Self) -> Callable[[ndarray], ndarray]:
        """Returns the composition of process applied over the dataset."""
        return self._process

    def standardize(self: Self) -> Self:
        """Standardizes the current dataset and stores the process."""
        standardizer = self._create_standardizer()
        if self._process == Preprocessor.identity:
            self._process = standardizer
        else:
            previous_process = self._process
            self._process = lambda x: standardizer(previous_process(x))
        self._data = standardizer(self._data)
        self._stat = Statistics(self._data).stats
        return self

    def normalize(self: Self) -> Self:
        """Normalizes the current dataset and stores the process."""
        normalizer = self._create_normalizer()
        if self._process == Preprocessor.identity:
            self._process = normalizer
        else:
            previous_process = self._process
            self._process = lambda x: normalizer(previous_process(x))
        self._data = normalizer(self._data)
        self._stat = Statistics(self._data).stats
        return self

    def _create_standardizer(self: Self) -> Callable[[ndarray], ndarray]:
        """Creates a standardizer for the current dataset."""
        mean = self._stat.loc["Mean"].to_numpy().copy()
        std = self._stat.loc["Std"].to_numpy().copy()

        def standardizer(x: ndarray | DataFrame) -> ndarray | DataFrame:
            return (x - mean) / std

        return standardizer

    def _create_normalizer(self: Self) -> Callable[[ndarray], ndarray]:
        """Creates a normalizer for the current dataset."""
        min = self._stat.loc["Min"].to_numpy().copy()
        scale = self._stat.loc["Max"].to_numpy() - min

        def normalizer(x: ndarray | DataFrame) -> ndarray | DataFrame:
            return (x - min) / scale

        return normalizer

    @staticmethod
    def identity(x: ndarray) -> ndarray:
        return x


def main() -> int:
    try:
        test = DataFrame([[1, 0, 0.8],
                          [4, 375, 7],
                          [9, -3, 12]])
        processor = Preprocessor(test)
        print("Data", test, sep="\n\n", end="\n\n")
        processor.standardize()
        print("Standaridzed", processor.data, sep="\n\n", end="\n\n")
        processor.normalize()
        print("Normalized", processor.data, sep="\n\n", end="\n\n")
        print("Reset", test, sep="\n\n", end="\n\n")
        print("Composition", processor.process(test), sep="\n\n", end="\n\n")
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
