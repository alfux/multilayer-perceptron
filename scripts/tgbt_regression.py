import argparse as arg
from argparse import Namespace
import sys
import traceback
from typing import Generator

import numpy as np
import pandas as pd

from Teacher import Teacher, MLP, Layer, Neuron


def get_args(description: str) -> Namespace:
    """Manages program arguments.

    Args:
        description is the program helper description.
    Returns:
        A Namespace of the arguments.
    """
    av = arg.ArgumentParser(description=description)
    help = "activate debug mode"
    av.add_argument("--debug", action="store_true", help=help)
    help = "number of epoch of training"
    av.add_argument("--epoch", type=int, help=help, default=1)
    help = "size of the samples to use at each epoch"
    av.add_argument("--sample", type=float, help=help, default=1)
    help = "path of the saved model file"
    av.add_argument("--save", default="default.npy", help=help)
    help = "name of a Neuron from the Neuron class"
    av.add_argument("--cost", default="MSE", help=help)
    LAYERS = [2, 32, 16, 8, 4, 2]
    help = "list of each layers size before the output layer"
    av.add_argument("--layers", default=LAYERS, nargs='*', type=int, help=help)
    help = "file with Timestamp, IEA, FaitJour, Temps (sec)"
    av.add_argument("file", help=help)
    return av.parse_args()


def gen_layers(layers: list[int], neuron: Neuron) -> Generator:
    """Generates an MLP based on hyperparameters"""
    bias = [Neuron(Neuron.bias, Neuron.dbias)]
    neuron = [neuron]
    for i in range(1, len(layers)):
        n = layers[i - 1] + 1
        matrix = np.random.randn(layers[i] + 1, n) * np.sqrt(2 / n)
        yield Layer(neuron * layers[i] + bias, matrix)
    yield Layer(neuron, np.random.randn(1, layers[i] + 1))


def main() -> int:
    """Creates a MLP regression of the TGBT current consumption over a day."""
    try:
        av = get_args(main.__doc__)
        data = pd.read_csv(av.file).loc[:, ["IEA", "Temps (sec)", "FaitJour"]]
        (lrelu, cost) = (Neuron("LReLU"), Neuron(av.cost))
        mlp = MLP(list(gen_layers([2, 32, 16, 8, 4, 2], lrelu)), cost)
        teacher = Teacher(data, "IEA", normal=[-1, 1], mlp=mlp)
        teacher.teach(av.epoch, time=True, frac=av.sample)
        sep = av.save[::-1].split('.', maxsplit=1)
        layers_fmt = "-".join([str(i) for i in av.layers])
        training_fmt = f"{layers_fmt}{av.cost}{av.epoch}x{av.sample}"[::-1]
        teacher.mlp.save(f"{training_fmt}_{sep[1]}"[::-1])
        return 0
    except Exception as err:
        if av.debug:
            print(traceback.format_exc())
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
