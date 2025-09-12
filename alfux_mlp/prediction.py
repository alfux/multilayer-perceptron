import argparse as arg
from argparse import Namespace
import logging
from pathlib import Path

from alfux_mlp import MLP
import pandas as pd


def get_args(description: str = "") -> Namespace:
    """Parse command-line arguments.

    Args:
        description (str): Program help description shown in ``--help``.

    Returns:
        Namespace: Parsed CLI arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("model", type=str, help="Path of the model's file.")
    av.add_argument("csv", type=str, help="Path of the dataset.")
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
    av.add_argument("--header", action="store_true", help="CSV has header.")
    av.add_argument("--drops", nargs='*', help="Columns to drop.", default=[])
    return av.parse_args()


def main() -> int:
    """Evaluate a saved model on a dataset.

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
        model = MLP.load(av.model)
        if av.header:
            df = pd.read_csv(av.csv).drop(av.drops, axis=1)
        else:
            av.drops = [int(x) for x in av.drops]
            df = pd.read_csv(av.csv, header=None).drop(av.drops, axis=1)
        df["Prediction"] = pd.Series(model.eval(df.to_numpy()))
        out = Path(av.csv)
        df.to_csv(
            out.with_name(out.stem + "_predicted.csv"), index=False,
            header=False
        )
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    main()
