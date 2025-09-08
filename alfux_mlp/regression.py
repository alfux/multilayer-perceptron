"""Program creating neural network based on JSON parameters."""
import argparse as arg
from argparse import Namespace
import json
import sys
import traceback
from typing import Generator

import numpy as np
import pandas as pd

from .teacher import Teacher, MLP, Layer, Neuron


def get_args(description: str = '') -> Namespace:
    """Manages program arguments.

    Args:
        ::description: is the program helper description.
    Returns:
        A Namespace of the arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_true", help="traceback mode")
    av.add_argument("json", help="Parameters file")
    return av.parse_args()


def gen_layers(layers: list[int], neuron: Neuron) -> Generator:
    """Generates an MLP based on hyperparameters"""
    bias = [Neuron("bias")]
    neuron = [neuron]
    for i in range(1, len(layers) - 1):
        n = layers[i - 1] + 1
        matrix = np.random.randn(layers[i] + 1, n) * np.sqrt(2 / n)
        yield Layer(neuron * layers[i] + bias, matrix)
    matrix = np.random.randn(layers[-1], layers[-2] + 1)
    yield Layer(neuron * layers[-1], matrix)


def main() -> int:
    """Uses MLP model to perform a regression.

    Configuration file as a JSON:
    \n{
    \t"file": "Path/to/the/trainning/file.csv",
    \t"truth": "Column name of the feature to predict/simulate",
    \t"layers": ["List of integers representing layers from input to output"],
    \t"epoch": "Number of trainning epoch as int",
    \t"sample": "Fraction of sample to use at each epoch as float",
    \t"outnorm": ["Two floats representing an interval for normalization"],
    \t"save": "path/to/save.npy",
    \t"n": "Number of model to generate",
    \t"activ": "Neuron function's name from the Neuron class",
    \t"cost": "Cost function's name from the Neuron class"
    }
    """
    try:
        av = get_args(main.__doc__)
        with open(av.json, "r") as file:
            cf = json.load(file)
        data = pd.read_csv(cf["file"]).drop(["Timestamp"], axis=1)
        (lrelu, cost) = (Neuron(cf["activ"]), Neuron(cf["cost"]))
        for i in range(cf["n"]):
            mlp = MLP(list(gen_layers(cf["layers"], lrelu)), cost)
            teacher = Teacher(data, cf["truth"], normal=cf["outnorm"], mlp=mlp)
            teacher.teach(cf["epoch"], time=True, frac=cf["sample"])
            sep = cf["save"][::-1].split('.', maxsplit=1)
            layers = "-".join([cf["activ"]] + [str(i) for i in cf["layers"]])
            layers = f"{layers}-{cf["cost"]}{cf["epoch"]}x{cf["sample"]}"[::-1]
            teacher.mlp.save(f"{layers}_{sep[1]}"[::-1] + f"_{i}")
        return 0
    except Exception as err:
        if "av" in locals() and hasattr(av, "debug") and av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
