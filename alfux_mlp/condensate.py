import argparse as arg
from argparse import Namespace
import os
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


def get_args(description: str = "") -> Namespace:
    """Parse command-line arguments.

    Args:
        description (str): Program help description shown in ``--help``.

    Returns:
        Namespace: Parsed CLI arguments.
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
    """Convert a value to ``float``.

    Args:
        x (Any): Value to convert.

    Returns:
        float: Converted value or ``nan`` on failure.
    """
    try:
        return float(x)
    except Exception:
        return float("nan")


def series_to_numeric(x: Series) -> Series:
    """Convert a pandas Series to numeric values.

    Non-convertible values become ``NaN``.

    Args:
        x (Series): Input series.

    Returns:
        Series: Numeric series.
    """
    x.apply(to_numeric)
    return x


def z_score(dataframe: DataFrame, threshold: float = 3.0) -> DataFrame:
    """Remove outliers using the z-score method.

    Args:
        dataframe (DataFrame): DataFrame to filter.
        threshold (float, optional): Z-score exclusion threshold
            (typically 2–3). Defaults to ``3.0``.

    Returns:
        DataFrame: Filtered DataFrame without detected outliers.
    """
    df = dataframe.copy()
    prev_shape = None
    while df.shape != prev_shape:
        prev_shape = df.shape
        df = df[(np.abs(df - df.mean()) / df.std() < threshold).all(axis=1)]
    return df


def main() -> int:
    """Concatenate multiple CSV files from a directory into one dataset.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
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
