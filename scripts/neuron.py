import argparse as arg
import sys
from typing import Self, Callable, Any

import pandas as pd
import numpy as np
from numpy import ndarray


class Neuron:
    """Neuron node for neural network."""

    def __init__(self: Self, weights: ndarray, function: Callable) -> None:
        """Waighted neuron is based on single parameter real function."""
        self._weights = weights
        self._activation_function = function
    
    def __call__(self: Self, input: ndarray) -> Any:
        """Computes neuron output."""
        
        self._activation_function


def main() -> None:
    """Displays neuron output from dataset input."""
    try:
        parser = arg.ArgumentParser(description=main.__doc__)
        parser.add_argument("data", help="csv dataset to compute")
        parser.add_argument("weights", help="csv weights to use")
        data = pd.read_csv(parser.parse_args().data)
        weights = pd.read_csv(parser.parse_args().weights)
        print(data, weights, sep="\n\n")

    except Exception as err:
        print(f"Error: {err.__class__.__name__}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
