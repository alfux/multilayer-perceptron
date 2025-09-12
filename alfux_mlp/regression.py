"""Create and train neural networks from JSON configurations."""
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
    """Train regressors on a dataset from a configuration.

    A configuration specifies the dataset, preprocessing, model layout, and
    training parameters.
    """

    def __init__(self: "Regression", config: dict) -> None:
        """Initialize from a configuration object.

        Args:
            config (dict): Configuration mapping. Keys include ``file``,
                ``header``, optional ``drops``, and other training settings.
        """
        if config["header"]:
            self._data = pd.read_csv(config["file"])
        else:
            self._data = pd.read_csv(config["file"], header=None)
        if "drops" in config:
            self._data = self._data.drop(config["drops"], axis=1)
        self._config = config

    def train(self: "Regression") -> MLP:
        """Train a model based on the loaded configuration.

        Returns:
            MLP: The trained model instance.
        """
        activ = Neuron(self._config["activ"])
        cost = Neuron(self._config["cost"])
        layers = list(Regression.gen_layers(self._config["layers"], activ))
        mlp = MLP(layers, cost, learning_rate=self._config["learning_rate"])
        teacher = Teacher(
            self._data, self._config["truth"], self._config["outnorm"], mlp,
            self._config.get("pre", None), self._config.get("post", None)
        )
        teacher.teach(self._config["epoch"], True, self._config["sample"])
        return teacher.mlp

    @staticmethod
    def gen_layers(layers: list[int], neuron: Neuron) -> Generator:
        """Yield untrained layers according to the configuration.

        Args:
            layers (list[int]): Layer sizes from input to output.
            neuron (Neuron): Activation function to use for hidden/output.

        Yields:
            Layer: Each constructed layer in order.
        """
        bias = [Neuron("bias")]
        neuron = [neuron]
        for i in range(1, len(layers) - 1):
            n = layers[i - 1] + 1
            matrix = np.random.randn(layers[i] + 1, n) * np.sqrt(2 / n)
            yield Layer(neuron * layers[i] + bias, matrix)
        matrix = np.random.randn(layers[-1], layers[-2] + 1)
        yield Layer(neuron * layers[-1], matrix)


def get_args(description: str = "") -> Namespace:
    """Parse command-line arguments.

    Args:
        description (str): Program help description shown in ``--help``.

    Returns:
        Namespace: Parsed CLI arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("--debug", action="store_true", help="traceback mode")
    av.add_argument("json", help="Parameters file")
    return av.parse_args()


def main() -> int:
    """Train and save MLP regressors defined in a JSON file.

    Examples:
        JSON configuration structure::

            {
                "file": "path/to/training.csv",
                "truth": "target_column_name",
                "drops": ["optional", "columns", "to", "drop"],
                "header": true,
                "layers": [n_input, hidden1, ..., n_output],
                "epoch": 100,
                "sample": 0.5,
                "outnorm": [0.0, 1.0],
                "save": "path/to/model.npy",
                "activ": "<Neuron activation name>",
                "cost": "<Neuron cost name>",
                "pre": "normalize|standardize",
                "post": "normalize|standardize|onehot"
            }

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
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
