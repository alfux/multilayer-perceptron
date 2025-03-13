import argparse as arg
from datetime import datetime
import sys
import traceback
from typing import Generator

import numpy as np
import pandas as pd
from pandas import DataFrame

from Teacher import Teacher, MLP, Layer, Neuron


def gen_layers(layers: list[int], neuron: Neuron) -> Generator:
    """Generates an MLP based on hyperparameters"""
    bias = [Neuron("Neuron.bias", "Neuron.dbias")]
    neuron = [neuron]
    for i in range(1, len(layers)):
        n = layers[i - 1] + 1
        matrix = np.random.randn(layers[i] + 1, n) * np.sqrt(2 / n)
        yield Layer(neuron * layers[i] + bias, matrix)
    yield Layer(neuron, np.random.randn(1, layers[i] + 1))


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
        data: DataFrame = pd.read_csv(av.file)
        data = data.loc[:, ["IEA", "Temps (sec)", "FaitJour"]]
        lrelu = Neuron("Neuron.LReLU", "Neuron.dLReLU")
        cost = Neuron("Neuron.MAE", "Neuron.dMAE")
        mlp = MLP(list(gen_layers([2, 32, 16, 8, 4, 2], lrelu)), cost)
        teacher = Teacher(data, "IEA", normal=[-1, 1], mlp=mlp)
        teacher.teach(av.epoch, time=True, frac=av.sample)
        sep = av.save[::-1].split('.', maxsplit=1)
        date = datetime.now().date().isoformat()[::-1]
        teacher.mlp.save((date + "_" + sep[1])[::-1])
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
