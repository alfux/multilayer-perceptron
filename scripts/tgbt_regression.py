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
        av.add_argument("--sample", type=float, help="sample frac", default=1)
        av.add_argument("--save", default="default.mlp", help="saving path")
        av.add_argument("file", help="file with timestamp and IEA")
        av = av.parse_args()
        data: DataFrame = pd.read_csv(av.file).sample(frac=av.sample)
        data = data.loc[:, ["IEA", "Temps (sec)", "FaitJour"]]
        lrelu = Neuron("Neuron.LReLU", "Neuron.dLReLU")
        bias = Neuron("Neuron.bias", "Neuron.dbias")
        (L1, L2, L3, L4) = (16, 8, 4, 2)
        mlp = MLP([
            Layer([lrelu] * L1 + [bias], np.random.randn(L1 + 1, 3)),
            Layer([lrelu] * L2 + [bias], np.random.randn(L2 + 1, L1 + 1)),
            Layer([lrelu] * L3 + [bias], np.random.randn(L3 + 1, L2 + 1)),
            Layer([lrelu] * L4 + [bias], np.random.randn(L4 + 1, L3 + 1)),
            Layer([lrelu], np.random.randn(1, L4 + 1)),
            ], Neuron("Neuron.MSE", "Neuron.dMSE")
        )
        Teacher(
            data, "IEA", normal=[-1, 1], mlp=mlp
        ).teach(av.epoch).save(av.save)
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
