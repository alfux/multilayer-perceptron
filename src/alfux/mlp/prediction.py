import argparse as arg
from argparse import Namespace
import logging
from pathlib import Path

from alfux.mlp import MLP
import pandas as pd
from pandas import Series


def load_data(path: Path, target: str, drops: list, header: bool) -> tuple:
    """Load data.

    Args:
        path (Path): Path of the CSV file.
        target (str): Target column.
        drop (list): Columns to drop.
        header (bool): Wether the csv file has a header or not.
    Returns:
        tuple: (dataframe, target)
    """
    if header:
        df = pd.read_csv(path).drop(drops, axis=1)
        if target is not None:
            target_list = df[target]
            df = df.drop([target], axis=1)
        else:
            target_list = None
    else:
        drops = [int(x) for x in drops]
        df = pd.read_csv(path, header=None).drop(drops, axis=1)
        if target is not None:
            target_list = df[int(target)]
            df = df.drop([int(target)], axis=1)
        else:
            target_list = None
    return df, target_list


def prediction_accuracy(target: Series, prediction: Series) -> float:
    """Compute prediction accuracy compared to the given set.

    Args:
        target (Series): Target value, the truth from the dataset.
        prediction (Series): The output of the MLP.
    Returns:
        float: Ratio of correct prediction.
    """
    if target is not None:
        accuracy = (target == prediction).sum() / len(target)
        accuracy = f"{accuracy * 100:.2f}% ({accuracy})"
        print("Accuracy by Binary Cross Entropy Loss function = ", accuracy)
        return accuracy
    else:
        print("Accuracy couldn't be computed by lack of target data.")
        return float("NaN")


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
    av.add_argument("--header", action="store_true", help="CSV has header.")
    av.add_argument("--drops", nargs='*', help="Columns to drop.", default=[])
    av.add_argument("--target", help="Target column.", default=None)
    av.add_argument("--debug", action="store_true", help="Traceback mode.")
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
        model = MLP.loadf(av.model)
        df, target = load_data(av.csv, av.target, av.drops, av.header)
        prediciton = Series(model.eval(df.to_numpy()))
        prediction_accuracy(target, prediciton)
        path = Path(av.csv)
        path = path.with_name(path.stem + "_predicted.csv")
        prediciton.to_csv(path, index=False, header=False)
        return 0
    except Exception as err:
        debug = "av" in locals() and hasattr(av, "debug") and av.debug
        logging.critical("Fatal error: %s", err, exc_info=debug)
        return 1


if __name__ == "__main__":
    main()
