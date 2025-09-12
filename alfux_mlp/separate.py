import argparse as arg
from argparse import Namespace
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame


def split(df: DataFrame, frac: float = 0.8) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame into train/validation parts.

    Args:
        df (DataFrame): Input DataFrame to split.
        frac (float, optional): Proportion of the first split (train). The
            second split uses ``1 - frac``. Defaults to ``0.8``.

    Returns:
        tuple[DataFrame, DataFrame]: The two splits ``(first, second)``.
    """
    df = df.to_numpy()
    frac = int(np.clip(frac, 0, 1) * len(df))
    indexes = np.random.permutation(len(df))
    return DataFrame(df[indexes[:frac]]), DataFrame(df[indexes[frac:]])


def get_args(description: str = "") -> Namespace:
    """Parse command-line arguments.

    Args:
        description (str): Program help description shown in ``--help``.

    Returns:
        Namespace: Parsed CLI arguments.
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
    """Split a CSV into two CSV files (default 80/20).

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
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
