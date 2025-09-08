import argparse as arg
from argparse import Namespace
import os
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        ::description: is the program helper description.
    Returns:
        A Namespace of the arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_true", help="debug mode")
    av.add_argument("path", help="directory containing all csv files")
    av.add_argument("--ignore", nargs='*', help="ignore feats", default=[])
    message = "columns for z-score process"
    av.add_argument("--z-score", nargs='*', help=message, default=[])
    av.add_argument("--save", default='default.csv', help="save file path")
    return av.parse_args()


def to_numeric(x: Any) -> float:
    """Converts x to a float."""
    try:
        return float(x)
    except Exception:
        return float("nan")


def series_to_numeric(x: Series) -> Series:
    """Converts a Series to numeric types"""
    x.apply(to_numeric)
    return x


def z_score(dataframe: DataFrame, threshold: float = 3.0) -> DataFrame:
    """Remove outliers with z-score method.

    Args:
        dataframe (DataFrame): A dataframe to filter
        threshold (float): The z-score exclusion threshold (2 or 3 typically)
    Returns:
        DataFrame: The filtered data.
    """
    df = dataframe.copy()
    prev_shape = None
    while df.shape != prev_shape:
        prev_shape = df.shape
        df = df[(np.abs(df - df.mean()) / df.std() < threshold).all(axis=1)]
    return df


def main() -> int:
    """Condensate multiple csv files of a directory into one set."""
    try:
        av = get_args(main.__doc__)
        data = [pd.read_csv(av.path + "/" + f) for f in os.listdir(av.path)]
        data = pd.concat(data).dropna(how="all", axis=1)
        data = data.reset_index(drop=True).apply(series_to_numeric)
        if len(av.z_score) > 0:
            z_sub = data[av.z_score]
            z_sub = z_score(z_sub)
            data = data.loc[z_sub.index]
            data[av.z_score] = z_sub
        data = data.loc[:, (data != data.iloc[0]).any(axis=0)]
        data.to_csv(av.save, index=False)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
