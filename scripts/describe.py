import argparse as arg
import sys
from typing import Self, Generator

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


class Statistics:
    """Computes different statistics on dataset."""

    def __init__(self: Self, data: DataFrame) -> None:
        """Process dataset"""
        self._fields = ["N", "Mean", "Var", "Std", "Min",
                        "25%", "50%", "75%", "Max"]
        self.stats = DataFrame(self._generate_stats(data.copy())).transpose()
        self.stats.columns = data.columns

    def _generate_stats(self: Self, data: DataFrame) -> Generator:
        """Iterable over stat Series computed from data."""
        for x in data.columns:
            yield self._compute(data[x])

    def _compute(self: Self, column: Series) -> Series:
        """Computes stats on the column's data and return it as a Series"""
        stat = [0] * len(self._fields)
        column = [x for x in column if x == x]
        list.sort(column)
        stat[0] = len(column)
        stat[1] = np.sum(column) / stat[0]
        stat[2] = np.sum((column - stat[1]) ** 2) / (stat[0] - 1)
        stat[3] = np.sqrt(stat[2])
        stat[4] = column[0]
        stat[5] = self._percentile(column, 25)
        stat[6] = self._percentile(column, 50)
        stat[7] = self._percentile(column, 75)
        stat[8] = column[-1]
        return Series(stat, self._fields)

    def _percentile(self: Self, sorted: list, p: float) -> float:
        """Computes percentile of sorted list."""
        p = (len(sorted) - 1) * np.clip(p, 0, 100) / 100
        r = p % 1
        p = int(p - r)
        if r != 0:
            return sorted[p] + r * (sorted[p + 1] - sorted[p])
        return sorted[p]


def main() -> None:
    """Prints a description of numerical values frome csv file."""
    try:
        parser = arg.ArgumentParser(description="Prints csv dataset stats")
        parser.add_argument("data", help="csv dataset")
        parser.add_argument("-n", "--no-header", action="store_true")
        if parser.parse_args().no_header:
            data = pd.read_csv(parser.parse_args().data, header=None)
        else:
            data = pd.read_csv(parser.parse_args().data)
        print(Statistics(data.select_dtypes([float, int])).stats)
    except Exception as err:
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
