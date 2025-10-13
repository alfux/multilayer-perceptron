import argparse as arg
import sys
from typing import Generator

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class Statistics:
    """Compute basic descriptive statistics for a dataset."""

    def __init__(self: "Statistics", data: DataFrame) -> None:
        """Compute statistics for the provided DataFrame.

        Args:
            data (DataFrame): Input data with numeric columns.
        """
        self._fields = ["N", "Mean", "Var", "Std", "Min",
                        "25%", "50%", "75%", "Max"]
        self.stats = DataFrame(self._generate_stats(data)).transpose()
        self.stats.columns = data.columns

    def _generate_stats(self: "Statistics", data: DataFrame) -> Generator:
        """Iterate over per-column statistics series."""
        for x in data.columns:
            yield self._compute(data[x])

    def _compute(self: "Statistics", column: Series) -> Series:
        """Compute statistics on a single column.

        Args:
            column (Series): Column to analyze.

        Returns:
            Series: Statistics with fields ``N, Mean, Var, Std, Min, 25%,
            50%, 75%, Max``.
        """
        stat = [0] * len(self._fields)
        column = column.to_numpy(dtype=float)
        column = np.sort(column[~np.isnan(column)])
        stat[0] = len(column)
        stat[1] = np.sum(column) / stat[0]
        stat[2] = ((np.sum(column ** 2) / stat[0]) - (stat[1] ** 2))
        stat[2] *= stat[0] / (stat[0] - 1)
        stat[3] = np.sqrt(stat[2])
        stat[4] = column[0]
        stat[5] = self._percentile(column, 25)
        stat[6] = self._percentile(column, 50)
        stat[7] = self._percentile(column, 75)
        stat[8] = column[-1]
        return Series(stat, self._fields)

    def _percentile(self: "Statistics", sorted: list, p: float) -> float:
        """Compute the percentile of a sorted list."""
        p = (len(sorted) - 1) * np.clip(p, 0, 100) / 100
        r = p % 1
        p = int(p - r)
        if r != 0:
            return sorted[p] + r * (sorted[p + 1] - sorted[p])
        return sorted[p]


def main() -> int:
    """Print descriptive statistics of numeric columns in a CSV file.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
    try:
        parser = arg.ArgumentParser(description=main.__doc__)
        parser.add_argument("data", help="csv dataset")
        parser.add_argument("-n", "--no-header", action="store_true")
        if parser.parse_args().no_header:
            data = pd.read_csv(parser.parse_args().data, header=None)
        else:
            data = pd.read_csv(parser.parse_args().data)
        print(Statistics(data.select_dtypes([float, int])).stats)
        return 0
    except Exception as err:
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
