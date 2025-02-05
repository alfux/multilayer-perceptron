import argparse
import sys
import traceback

import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame


def main() -> int:
    try:
        av = argparse.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="Debug mode")
        av.add_argument("path", help="Path to the csv file")
        av = av.parse_args()
        df: DataFrame = pd.read_csv(av.path, sep=';', decimal=',')
        df = df.dropna(how="all", axis=1).dropna(how="any")
        df.reset_index(drop=True, inplace=True)
        df = df.loc[:,~(df == df.loc[0]).all(axis=0)]
        df = df.drop_duplicates(subset=df.columns[1:]).reset_index(drop=True)
        tags: ndarray = np.array(["+", "-", "="])
        rq: ndarray = (np.round(np.random.randn(df.shape[0])) % 3).astype(int)
        df = pd.concat([DataFrame({"Etat": tags[rq]}), df], axis=1)
        df.to_csv(av.path[:-4] + "_random_qualified.csv", index=False)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()