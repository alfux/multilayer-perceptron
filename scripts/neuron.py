import sys
from typing import Self, Callable

import numpy as np
from numpy import ndarray


class Neuron:
    """Neuron node for a neural network."""

    def __init__(self: Self, f: Callable = None, df: Callable = None) -> None:
        """Define neuron with its activation function and derivative.

        Args:
            <f> and <df> must be a C1 single parameter real function and it's
            derivative, respectively.
        """
        if f is None:
            self._f = lambda x: x
            self._df = lambda x: 1
        elif df is None:
            self._f = f
            self._df = Neuron._finite_diff
        else:
            self._f = f
            self._df = df

    def __call__(self: Self, x: float | ndarray) -> float | ndarray:
        """Computes neuron's output."""
        return self._f(x)

    def deriv(self: Self, x: float | ndarray) -> float | ndarray:
        """Computes neuron's derivative output."""
        return self._df(x)

    def _finite_diff(self: Self, x: float | ndarray) -> float | ndarray:
        """Finite differences replaces the derivative if none is provided."""
        return (self._f(x + 1e-6) - self._f(x - 1e-6)) / (2 * 1e-6)


def main() -> None:
    """Displays neuron output from dataset input."""
    try:
        print("Neuron presentation:")
        end = False
        while not end:
            try:
                funct = eval(input("\tfunct: "))
                deriv = eval(input("\tderiv: "))
                neuron = Neuron(funct, deriv)
                value = np.random.rand(1)
                print(f"\t\tfunc({value}) = {neuron(value)}")
                print(f"\t\tfunc.deriv({value}) = {neuron.deriv(value)}")
            except Exception as err:
                print(f"Error: {type(err).__name__}: {err}", file=sys.stderr)
            finally:
                end = input("Continue ? (y/n): ").casefold() in ('n', "no")
        return 0
    except Exception as err:
        print(f"Error: {type(err).__name__}: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
