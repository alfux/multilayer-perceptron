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


class Regression:
    """Regressions on a dataset based on a configuration file."""

    def __init__(self: "Regression", config: dict) -> None:
        """Initialize a Regression object based on a config JSON object.

        Args:
            self ("Regression"): The current instance.
            config (dict): The configuration object.
        Returns:
            None.
        """
        if config["header"]:
            self._data = pd.read_csv(config["file"])
        else:
            self._data = pd.read_csv(config["file"], header=None)
        if "drops" in config:
            self._data = self._data.drop(config["drops"], axis=1)
        self._config = config

    def train(self: "Regression") -> MLP:
        """Generate a trained model based on the loaded configuration.

        Args:
            self ("Regression"): The current instance.
        Returns:
            MLP: The trained model.
        """
        activ = Neuron(self._config["activ"])
        cost = Neuron(self._config["cost"])
        layers = list(Regression.gen_layers(self._config["layers"], activ))
        mlp = MLP(layers, cost)
        teacher = Teacher(
            self._data, self._config["truth"], self._config["outnorm"], mlp,
            self._config.get("pre", None), self._config.get("post", None)
        )
        teacher.teach(self._config["epoch"], True, self._config["sample"])
        return teacher.mlp

    @staticmethod
    def gen_layers(layers: list[int], neuron: Neuron) -> Generator:
        """Generate an untrained MLP object based on configuration.

        Args:
            layers (list[int]): A list representing each layers.
            neuron (Neuron): The activation function of the neural network.
        Yields:
            Layer: The last constructed layer.
        """
        bias = [Neuron("bias")]
        neuron = [neuron]
        for i in range(1, len(layers) - 1):
            n = layers[i - 1] + 1
            matrix = np.random.randn(layers[i] + 1, n) * np.sqrt(2 / n)
            yield Layer(neuron * layers[i] + bias, matrix)
        matrix = np.random.randn(layers[-1], layers[-2] + 1)
        yield Layer(neuron * layers[-1], matrix)


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


def main() -> int:
    """Uses MLP model to perform a regression.

    Configuration file as a JSON:
    \n{
    \t"file": "Path/to/the/trainning/file.csv",
    \t"truth": "Column name of the feature to predict/simulate",
    \t"drops": "Optionally drop features from the dataset.",
    \t"layers": ["List of integers representing layers from input to output"],
    \t"epoch": "Number of trainning epoch as int",
    \t"sample": "Fraction of sample to use at each epoch as float",
    \t"outnorm": ["Two floats representing an interval for normalization"],
    \t"save": "path/to/save.npy",
    \t"activ": "Neuron function's name from the Neuron class",
    \t"cost": "Cost function's name from the Neuron class"
    \t"pre": "type of data regularization (normalize/standardize)"
    \t"post": "type of data post processing (normalize/standardize/onehot)"
    }
    """
    try:
        av = get_args(main.__doc__)
        with open(av.json, "r") as file:
            configurations = json.load(file)
        for config in configurations:
            regressor = Regression(config)
            regressor.train().save(config["save"])
        return 0
    except Exception as err:
        if "av" in locals() and hasattr(av, "debug") and av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
