import argparse as arg
from argparse import Namespace
from collections import deque
from datetime import datetime
import json
import sys
import traceback

import numpy as np
from numpy import ndarray
import pandas as pd

from .display import Display
from .mlp import MLP
from .processor import Processor


class Teacher:
    """Train an MLP using provided data and configuration."""

    class BadTeacher(Exception):
        """Teacher-specific exception type."""
        pass

    def __init__(self: "Teacher", config: dict) -> None:
        """Create a Teacher for an MLP.

        Args:
            config (dict): Hyperparameters describing how the model trains.
        """
        if config["header"]:
            self._data = pd.read_csv(config["data"])
            self._vdata = pd.read_csv(config["validation"])
        else:
            self._data = pd.read_csv(config["data"], header=None)
            self._vdata = pd.read_csv(config["validation"], header=None)
        if "drops" in config:
            self._data = self._data.drop(config["drops"], axis=1)
            self._vdata = self._vdata.drop(config["drops"], axis=1)
        with open(config["model"]) as file:
            model = json.loads(file.read())
        self._mlp: MLP = MLP.loadd(model)
        self._process(config, model)
        self._data: ndarray = self._proc.data
        self._vdata: ndarray = self._proc.vdata
        self._target: ndarray = self._proc.target
        self._vtarget: ndarray = self._proc.vtarget
        self._config = config
        self._display = Display(2, **self._config["display"])
        self._t = datetime.now()

    @property
    def mlp(self: "Teacher") -> MLP:
        """Get the underlying MLP model."""
        return self._mlp

    @mlp.setter
    def mlp(self: "Teacher", value: MLP) -> None:
        """Set the underlying MLP model."""
        self._mlp = value

    def teach(self: "Teacher") -> "Teacher":
        """Train the internal MLP for a number of epochs.

        Returns:
            Teacher: The current instance.
        """
        if self._mlp is None:
            raise Teacher.BadTeacher("No MLP loaded.")
        self._t = datetime.now()
        if self._config["display"]:
            self._mlp.preprocess = []
            self._mlp.postprocess = []
            y, x = self._sample(self._config["frac"])
            self._display(self._mlp.cost.eval(y, self._mlp.eval(x))[0], 1)
        cost = float("+inf")
        for i in range(self._config["epoch"]):
            print(f"\nEpoch {i}:")
            prev = cost
            cost = self._epoch()
            if self._early_stopping(i, prev, cost):
                self._mlp.revert()
                break
        self._mlp.preprocess = self._proc.preprocess
        self._mlp.postprocess = self._proc.postprocess
        self._training_time = datetime.now() - self._t
        if self._config["time"]:
            print("\n\tTraining time:", self._training_time)
        return self

    def _early_stopping(
            self: "Teacher", i: int, prev: float, cost: float
    ) -> bool:
        """Stop the algorithm if needed.

        Args:
            i (int): Epoch iteration.
            prev (float): The previous cost.
            cost (float): The current cost.
        Returns:
            bool: True if the previous cost is lower than the current one.
        """
        return (
            "early_stopping" in self._config
            and i > self._config["early_stopping"]
            and prev < cost
        )

    def _epoch(self: "Teacher") -> float:
        """Perform an epoch of the training routine.

        Returns:
            float: The cost at the end of the epoch.
        """
        if self._config["display"]:
            total, i = 0, 0
            for value in self._mlp.update(*self._sample(self._config["frac"])):
                self._display(value[0], 0)
                total += value[0]
                i += 1
            cost = total / i
            self._display(cost, 1)
            self._display_metrics(cost)
            return cost
        else:
            cost = deque(self._mlp.update(*self._sample(self._config["frac"])))
            return np.sum(cost) / len(cost)

    def _display_metrics(self: "Teacher", cost: float) -> None:
        """Displays metrics during the training.

        Args:
            cost (float): loss value over the dataset at the last update.
        """
        vloss, accuracy = self._validation()
        metrics = {
            "Time": datetime.now() - self._t,
            "Data loss": cost,
            "Gradient norm": self._mlp.last_gradient_norm,
            "Validation loss": vloss,
            "Validation accuracy": accuracy
        }
        self._display.metrics(**metrics)

    def _validation(self: "Teacher") -> list:
        """Compute validation loss.

        Returns:
            list: [validation loss, accuracy]
        """
        out = self._mlp.eval(self._vdata)
        vloss = self._mlp.cost.eval(self._vtarget, out)
        accuracy = (self._vtarget == out).all(axis=1).sum() / out.shape[0]
        return vloss, accuracy

    def _process(self: "Teacher", config: dict, model: list) -> None:
        """Process datas.

        Args:
            config (dict): Configuration.
            model (list): Model description.
        """
        self._proc = Processor(self._data, config["target"], valid=self._vdata)
        for pre in model["preprocess"]:
            match pre["activation"]:
                case "normalize":
                    self._proc.pre_normalize(pre["parameters"][2])
                case "standardize":
                    self._proc.pre_standardize()
                case "add_bias":
                    self._proc.pre_bias()
                case _:
                    raise Teacher.BadTeacher("_process: Unknown parameter")
        for post in model["postprocess"]:
            match post["activation"]:
                case "unrmalize":
                    self._proc.post_normalize(post["parameters"][2])
                case "unstdardize":
                    self._proc.post_standardize()
                case "revonehot":
                    self._proc.onehot()
                case _:
                    raise Teacher.BadTeacher("_process: Unknown parameter")

    def _sample(self: "Teacher", frac: float) -> tuple[ndarray, ndarray]:
        """Select a random sample of the data.

        Args:
            frac (float): Proportion of the dataset to sample.

        Returns:
            tuple[ndarray, ndarray]: ``(truth, data)`` mini-batch.
        """
        size = np.int64(np.round(np.clip(frac, 0, 1) * self._data.shape[0]))
        idx = np.random.choice(self._data.shape[0], size=size, replace=False)
        return (self._target[idx], self._data[idx])


def get_args(description: str = "") -> Namespace:
    """Parse command-line arguments.

    Args:
        description (str): Program help description shown in ``--help``.

    Returns:
        Namespace: Parsed CLI arguments.
    """
    av = arg.ArgumentParser(description=description)
    av.add_argument("json", help="Parameters file")
    av.add_argument("--debug", action="store_true", help="traceback mode")
    return av.parse_args()


def main() -> int:
    """Train and save MLP regressors defined in a JSON file.

    Examples:
        JSON configuration structure:\n
            [
                {
                  "data": "./datas/data_training.csv",
                  "header": false,
                  "target": 1,
                  "drops": [0],
                  "epoch": 5,
                  "frac": 0.8,
                  "display": true,
                  "time": true,
                  "model": "./model.json",
                  "learning_rate": 1e-3,
                  "save": "./model_trained.json"
                },
                ...
            ]
    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
    try:
        av = get_args(main.__doc__)
        with open(av.json, "r") as file:
            configurations = json.load(file)
        for config in configurations:
            Teacher(config).teach().mlp.save(config["save"])
        Display.pause()
        return 0
    except Exception as err:
        if "av" in locals() and hasattr(av, "debug") and av.debug:
            print(traceback.format_exc(), file=sys.stderr)
        print(f"\n\tFatal: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
