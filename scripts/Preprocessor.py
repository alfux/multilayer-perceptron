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
        self._target: DataFrame = dataset.loc[:, [target]]
        self._unique: ndarray = None
        self._data: DataFrame = dataset.drop([target], axis=1)
        constant = (self._data == self._data.iloc[0]).all(axis=0)
        self._data = self._data.loc[:, ~constant]
        self._stat: DataFrame = Statistics(self._data).stats
        self._preprocess: Callable = Preprocessor.identity
        self._prestr = "lambda x: x"
        self._postprocess: Callable = Preprocessor.identity
        self._poststr = "lamnda x: x"

    @property
    def prestr(self: Self) -> str:
        """String of the input transformation. Can be used with eval()."""
        return self._prestr

    @property
    def poststr(self: Self) -> str:
        """"String of the output transformation. Can be used with eval(). """
        return self._poststr

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
    def unique(self: Self) -> ndarray:
        """Getter of the ordered list of unique labels."""
        return self._unique

    @property
    def data(self: Self) -> ndarray:
        """Getter of the current unprocessed dataset."""
        return self._data.to_numpy()

    @data.setter
    def data(self: Self, value: DataFrame | ndarray) -> None:
        """Setter of the current dataset."""
        if isinstance(value, ndarray):
            self._data = DataFrame(value)
        else:
            self._data = value
        self._stat = Statistics(self._data).stats

    def pre(self: Self, x: ndarray) -> ndarray:
        "Calling preprocessor."
        return self._preprocess(x)

    def post(self: Self, x: ndarray) -> ndarray:
        "Calling postprocessor"
        return self._postprocess(x)

    def onehot(self: Self) -> Self:
        """Transform the label field in a vectorized equivalent."""
        if self._unique is None:
            (uni, inv) = np.unique(self._target, return_inverse=True)
            self._target = np.eye(len(uni))[inv]
            self._unique = uni
        return self

    def post_standardize(self: Self) -> Self:
        """Standardizes the current target and stores the inverse process.

        <b>Returns:</b>
            The current instance of the class.
        """
        stats = Statistics(self._target).stats
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        mean = stats.loc["Mean"].to_numpy()
        std = stats.loc["Std"].to_numpy()
        strepr = f"lambda x: (x - np.{repr(mean)}) / np.{repr(std)})"
        strrev = f"lambda x: np.{repr(std)} * x + np.{repr(mean)}"
        np.set_printoptions()
        return self._post_apply(strepr, strrev)

    def post_normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current target and stores the inverse process.

        <b>Returns:</b>
            The current instance of the class.
        """
        stats = Statistics(self._target).stats
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        m = stats.loc["Min"].to_numpy()
        s = stats.loc["Max"].to_numpy() - m
        rep = f"{b[0]} + {b[1] - b[0]} * (x - np.{repr(m)}) / np.{repr(s)}"
        rev = f"np.{repr(m)} + np.{repr(s)} * (x - {b[0]}) / {b[1] - b[0]}"
        np.set_printoptions()
        return self._post_apply("lambda x: " + rep, "lambda x: " + rev)

    def pre_standardize(self: Self) -> Self:
        """Standardizes the current dataset and stores the process.

        <b>Returns:</b>
            The current instance of the class.
        """
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        mean = self._stat.loc["Mean"].to_numpy()
        std = self._stat.loc["Std"].to_numpy()
        strepr = f"lambda x: (x - np.{repr(mean)}) / np.{repr(std)})"
        np.set_printoptions()
        return self._pre_apply(strepr)

    def pre_normalize(self: Self, b: list[float] = [0, 1]) -> Self:
        """Normalizes the current dataset and stores the process.

        Args:
            <b> represents the wanted interval.
        Returns:
            The current instance of the class.
        """
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        m = self._stat.loc["Min"].to_numpy()
        s = self._stat.loc["Max"].to_numpy() - m
        rep = f"{b[0]} + {b[1] - b[0]} * (x - np.{repr(m)}) / np.{repr(s)}"
        np.set_printoptions()
        return self._pre_apply("lambda x: " + rep)

    def add_bias(self: Self) -> Self:
        """Adds a bias component at the end of the vector."""
        return self._pre_apply("Preprocessor.adding_bias")

    def _pre_apply(self: Self, strepr: str) -> Self:
        """Apply given function to the data and stores it.

        Args:
            <func> function to apply.
            <strepr> is a string representation of the added preprocess.
        """
        func = eval(strepr)
        if self._preprocess == Preprocessor.identity:
            self._preprocess = func
            self._prestr = strepr
        else:
            pre_func = self._preprocess
            self._preprocess = lambda x: func(pre_func(x))
            self._prestr = self._str_compose(strepr, self._prestr)
        self.data = func(self._data)
        return self

    def _post_apply(self: Self, strepr: str, strrev: str) -> Self:
        """Apply given function to the target and stores the invers.

        <b>Args:</b>
            <i>strepr</i> is the string to be eval() as the function.
            <i>strrev</i> is the string to be eval() as the inverse."""
        func = eval(strepr)
        if self._postprocess == Preprocessor.identity:
            self._postprocess = func
            self._poststr = strrev
        else:
            pre_func = self._postprocess
            self._postprocess = lambda x: pre_func(func(x))
            self._poststr = self._str_compose(self._poststr, strrev)
        self.target = func(self._target)
        return self

    def _str_compose(self: Self, outer: str, inner: str) -> str:
        """Compose the new str representation with the old one.
        Args:
            The new added process as a string.
        Returns:
            The composition as a string.
        """
        if ':' in outer:
            (prototype, body) = outer.split(':')
            (prefix, suffix) = body.split('x')
            return prototype + prefix + f"({inner})(x)" + suffix
        else:
            return f"lambda x: {outer}(({inner})(x))"

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
