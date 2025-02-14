import argparse as arg
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from MLP import MLP


def main() -> int:
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("mlp", help="path to mlp file")
        av.add_argument("baseline", help="path to the baseline")
        av = av.parse_args()
        mlp: MLP = MLP.load(av.mlp)
        baseline = pd.read_csv(av.baseline)
        plt.plot(baseline.iloc[:, 0], baseline.iloc[:, 1])
        prediction = mlp.eval(np.atleast_2d(baseline.iloc[:, 0].to_numpy()).T)
        plt.plot(baseline.iloc[:, 0], prediction)
        plt.legend(["TGBT", "MLP"])
        plt.show()
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
