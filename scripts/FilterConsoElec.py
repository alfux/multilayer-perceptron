import argparse as arg
import sys
from typing import Self
import traceback

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from Statistics import Statistics


class FilterConsoElec:
    """Filters electric consommation dataset."""

    def __init__(self: Self, data: DataFrame, **kwargs: dict) -> None:
        """Filters and separates dataframe in two for am and pm datas."""
        data = self._format_values(data)
        data = self._standardize(data, **kwargs)
        (self._am, self._pm) = self._separate(data)
        self.filtered = self._regroup(self._am, self._pm)

    def _format_values(self: Self, data: DataFrame) -> DataFrame:
        """Replaces column labels and values."""
        data.columns = ["Date", "Day", "Week", "Year", "AM", "PM", " ",
                        "D1 AM", "D1 PM", "D2 AM", "D2 PM", "P1 AM", "P1 PM",
                        "P2 AM", "P2 PM", "P3 AM", "P3 PM", "P4 AM", "P4 PM"]
        data = data.drop(["Date", "Day", "Year", "AM", "PM", " "], axis=1)
        data["Week"] = data["Week"].apply(lambda x: 'o' if x % 2 else 'e')

        def to_float(x: Series) -> Series:
            """Converts serie's values from string to float and returns it."""
            for i in x.index:
                try:
                    x[i] = float(str(x[i])[::-1].replace(',', '.', 1)[::-1])
                except Exception:
                    x[i] = float("nan")
            return x

        data.loc[:, "D1 AM":] = data.loc[:, "D1 AM":].apply(to_float)
        return data

    def _standardize(self: Self, data: DataFrame, **kwargs: dict) -> DataFrame:
        """Standardizes am and pm dataframes."""

        def standardize(data: Series) -> Series:
            """Standardizes a Series"""
            stats = Statistics(DataFrame([data]).transpose()).stats
            mean = stats.at["Mean", data.name]
            std = stats.at["Std", data.name]
            for i in data.index:
                if np.abs(data[i] - mean) > 2 * std:
                    data[i] = float("nan")
            return data

        data = data.loc[data.loc[:, "D1 AM":].dropna(axis=0, how="all").index]
        iter = kwargs["iter"] if "iter" in kwargs else 1
        for i in range(iter):
            data.loc[:, "D1 AM":] = data.loc[:, "D1 AM":].apply(standardize)
        return data

    def _separate(self: Self, data: DataFrame) -> tuple[DataFrame, DataFrame]:
        """Separates AM and PM values in two dataframes."""
        am = data.loc[:, "D1 AM"::2].dropna(axis=0, how='all')
        am.rename({"D1 AM": "D1", "D2 AM": "D2", "P1 AM": "P1", "P2 AM": "P2",
                  "P3 AM": "P3", "P4 AM": "P4"}, axis=1, inplace=True)
        pm = data.loc[:, "D1 PM"::2].dropna(axis=0, how='all')
        pm.rename({"D1 PM": "D1", "D2 PM": "D2", "P1 PM": "P1", "P2 PM": "P2",
                  "P3 PM": "P3", "P4 PM": "P4"}, axis=1, inplace=True)
        return (pd.concat([data["Week"].loc[am.index], am], axis=1),
                pd.concat([data["Week"].loc[pm.index], pm], axis=1))

    def _regroup(self: Self, am: DataFrame, pm: DataFrame) -> DataFrame:
        """Regroup values by teams."""
        for i in am.index:
            am.at[i, "Week"] = "A" if am.at[i, "Week"] == 'e' else "B"
        for i in pm.index:
            pm.at[i, "Week"] = "A" if pm.at[i, "Week"] == 'o' else "B"
        am.rename({"Week": "Team"}, axis=1, inplace=True)
        pm.rename({"Week": "Team"}, axis=1, inplace=True)
        data = pd.concat([am, pm], axis=0)
        return data.reset_index(drop=True)


def main() -> None:
    """Prints and filters dataset."""
    try:
        parser = arg.ArgumentParser()
        parser.add_argument("file", help="data to filter")
        parser.add_argument("-i", "--iterations", type=int, default=1,
                            help="number of iterations")
        data = pd.read_csv(parser.parse_args().file, header=None, sep=';')
        filter = FilterConsoElec(data, iter=parser.parse_args().iterations)
        print("\nFiltered datas:\n")
        print(filter.filtered)
        print("\n-> Printed in datas/data_teams.csv\n")
        filter.filtered.to_csv("datas/data_teams.csv", index=False)
    except Exception as err:
        print(traceback.format_exc())
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
