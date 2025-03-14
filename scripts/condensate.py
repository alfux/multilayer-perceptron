import argparse as arg
import os
import sys
import traceback

import pandas as pd


def main() -> int:
    """Condensate multiple csv files of a directory into one set."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("path", help="directory containing all csv files")
        av.add_argument("--ignore", nargs='*', help="ignore feats", default=[])
        av.add_argument("--save", default='default.csv', help="save file path")
        av = av.parse_args()
        data = [pd.read_csv(av.path + "/" + f) for f in os.listdir(av.path)]
        data = pd.concat(data).dropna(how="all", axis=1)
        data = data.reset_index(drop=True)
        relevant = data.drop(av.ignore, axis=1)
        relevant = relevant.apply(pd.to_numeric, errors="coerce")
        ignored = data.loc[:, av.ignore]
        relevant = relevant.dropna(how="any")
        ignored = ignored.loc[relevant.index].reset_index(drop=True)
        relevant = relevant.reset_index(drop=True)
        relevant = relevant.loc[:, ~(relevant == relevant.iloc[0]).all(axis=0)]
        pd.concat([ignored, relevant], axis=1).to_csv(av.save, index=False)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
