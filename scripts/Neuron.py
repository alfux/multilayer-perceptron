import sys
from typing import Self, Callable

import numpy as np
from numpy import ndarray


class Neuron:
    """Neuron node for a neural network.

    It is an object with real function and it's derivative.

    This class is expected to be used with full knowledge of how it works.
    There isn't any verification or security of any kind outside of __init__
    in order to reduce time and operations complexities.
    """

    def __init__(self: Self, f: Callable, df: Callable) -> None:
        """Define neuron with its activation function and derivative.

        Args:
            <f> and <df> must be a C1 single parameter real function and it's
            derivative, respectively.
        """
        self.eval: Callable = f
        self.diff: Callable = df

    def __repr__(self: Self) -> str:
        """String representation of the object."""
        return f"{self.eval.__name__},{self.diff.__name__}"

    def eval(self: Self, *args: list) -> ndarray:
        """Computes neuron's output.

        Args:
            <*args> the function is dynamically allocated and takes same
            arguments as the given function.
        Returns:
            The function's output as an ndarray
        """
        pass

    def diff(self: Self, *args: list) -> ndarray:
        """Derivative of the neuron. Computes differential in point x.

        Args:
            <*args> the function is dynamically allocated and takes same
            arguments as the given function.
        Returns:
            The function's output as an ndarray
        """
        pass

    def save(self: Self) -> Self:
        """Saves the current neuron."""
        print(f"Neuron({}, {})")
        return self


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
                print(f"\t\tfunc({value}) = {neuron.eval(value)}")
                print(f"\t\tfunc.deriv({value}) = {neuron.diff(value)}")
                neuron.save()
            except Exception as err:
                print(f"Error: {type(err).__name__}: {err}", file=sys.stderr)
            finally:
                end = input("Continue ? (y/n): ").casefold() in ('n', "no")
        return 0
    except Exception as err:
        print(f"\n\tError: {type(err).__name__}: {err}\n", file=sys.stderr)
        return 1


if __name__ == "__main__":
    main()
