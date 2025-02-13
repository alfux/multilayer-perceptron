import argparse as arg
import sys
import traceback

import numpy as np
import pandas as pd
from pandas import DataFrame

from Teacher import Teacher, MLP, Layer, Neuron


def main() -> int:
    """Creates a MLP regression of the TGBT current consumption over a day."""
    try:
        av = arg.ArgumentParser(description=main.__doc__)
        av.add_argument("--debug", action="store_true", help="debug mode")
        av.add_argument("--epoch", type=int, help="number of epoch", default=1)
        av.add_argument("file", help="file with timestamp and IEA")
        av = av.parse_args()
        data: DataFrame = pd.read_csv(av.file).loc[:, ["IEA", "Temps (sec)"]]
        lrelu = Neuron("Neuron.LReLU", "Neuron.dLReLU")
        bias = Neuron("Neuron.bias", "Neuron.dbias")
        mlp = MLP([
            Layer([lrelu] * 64 + [bias], np.random.randn(65, 2)),
            Layer([lrelu] * 32 + [bias], np.random.randn(33, 65)),
            Layer([lrelu] * 16 + [bias], np.random.randn(17, 33)),
            Layer([lrelu] * 8 + [bias], np.random.randn(9, 17)),
            Layer([lrelu], np.random.randn(1, 9)),
            ], Neuron("Neuron.MSE", "Neuron.dMSE")
        )
        Teacher(data, "IEA", normal=[-1, 1], mlp=mlp).teach(av.epoch).save()
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
