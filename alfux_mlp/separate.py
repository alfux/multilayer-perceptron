import argparse as arg
from argparse import Namespace
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame


def split(df: DataFrame, frac: float = 0.8) -> tuple[DataFrame, DataFrame]:
    """Split a df in two DataFrames with frac and 1 - frac proportions.

    Args:
        df (DataFrame): The DataFrame to split.
        frac (float): The proportion of the first part. The other part will
            be 1 - frac.
    Returns:
        tuple: The two DataFrames.
    """
    df = df.to_numpy()
    frac = int(np.clip(frac, 0, 1) * len(df))
    indexes = np.random.permutation(len(df))
    return DataFrame(df[indexes[:frac]]), DataFrame(df[indexes[frac:]])


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        description (str): is the program helper description.
    Returns:
        Namespace: The arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("csv", help="The path of the CSV file to split.")
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    av.add_argument("--header", action="store_true", help="CSV has header")
    av.add_argument(
        "--frac", type=float, help="fraction of the training set", default=0.8
    )
    return av.parse_args()


def main() -> int:
    """Separates a data CSV file in two CSV files with a 80/20 ratio.

    Returns:
        int: 1 for errors and 0 otherwise.
    """
    try:
        av = get_args(main.__doc__)
        FORMAT = "%(asctime)s | %(levelname)s - %(message)s"
        if av.debug:
            logging.basicConfig(level=logging.DEBUG, format=FORMAT)
        else:
            logging.basicConfig(level=logging.INFO, format=FORMAT)
        path = Path(av.csv)
        trn, vld = split(
            pd.read_csv(path) if av.header else pd.read_csv(path, header=None),
            av.frac
        )
        trn.to_csv(
            path.with_name(path.stem + "_training.csv"), index=False,
            header=False
        )
        vld.to_csv(
            path.with_name(path.stem + "_validation.csv"), index=False,
            header=False
        )
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    main()
