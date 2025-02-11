import argparse as arg
import sys
import traceback

import numpy as np
import pandas as pd
from pandas import DataFrame


def integrate_kW(data: DataFrame) -> float:
    """Computes the integral."""
    data.iloc[:, 3] = np.sqrt(3) * 428 * data.iloc[:, 3] * 0.5
    data.iloc[:, 1] = data.iloc[:, 1].diff() / 3600
    data.iloc[0, 1] = 0
    return np.sum(data.iloc[:, 1] * data.iloc[:, 3]) / 1000


def main() -> int:
    """Computes integral of a given timestamped current."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("file", help="formated verif file")
        av.add_argument("--debug", help="debug mode", action="store_true")
        av = av.parse_args()
        data = pd.read_csv(av.file)
        print(f"Last TGBT read: {data.iloc[-1, 2]:.2f} kWh")
        print(f"Computed integral: {integrate_kW(data):.2f} kWh")
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
