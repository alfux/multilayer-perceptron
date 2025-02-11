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

    def __init__(self: Self, f: str, df: str) -> None:
        """Define neuron with its activation function and derivative.

        Args:
            <f> and <df> must be a C1 single parameter real function and it's
            derivative, respectively. They are entered as a string that will be
            evaluated.
        """
        self._f: str = f
        self._df: str = df
        self.eval: Callable = eval(f)
        self.diff: Callable = eval(df)

    def __repr__(self: Self) -> str:
        """String representation of the object."""
        return f"Neuron(\"{self._f}\", \"{self._df}\")"

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

    @staticmethod
    def ReLU(x: ndarray) -> ndarray:
        """Computes the value of Rectified Linear Unit function in <x>."""
        return (x > 0).astype(float) * x

    @staticmethod
    def dReLU(x: ndarray) -> ndarray:
        """Computes the differential of ReLU in <x>."""
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x: float) -> float:
        """Computes the value of sigmoid function in <x>."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def dsigmoid(x: float) -> float:
        """Computes the differential of sigmoid in <x>."""
        return 1 / ((1 + np.exp(-x)) * (1 + np.exp(x)))

    @staticmethod
    def softmax(x: ndarray) -> ndarray:
        """Computes the value of softmax function in <x>."""
        out = np.exp(x)
        if x.ndim > 1:
            return out / np.atleast_2d(np.sum(out, axis=1)).T
        return out / np.sum(out)

    @staticmethod
    def dsoftmax(x: ndarray) -> ndarray:
        """Compute the differential of softmax in <x>."""
        vector = np.exp(x)
        vector /= np.sum(vector)
        Jacobian = -np.outer(vector, vector)
        index = np.arange(len(x))
        Jacobian[index, index] += vector
        return Jacobian

    @staticmethod
    def CELF(y: ndarray, x: ndarray) -> ndarray:
        """Cross Entropy Loss Function."""
        y = np.atleast_2d(y)
        x = np.atleast_2d(x)
        return -np.sum(np.log(np.einsum("ij,ij->i", y, x))) / x.shape[0]

    @staticmethod
    def dCELF(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Cross Entropy Loss Function."""
        y = np.atleast_2d(y)
        x = np.atleast_2d(x)
        return -np.sum(y / (x + 1e-15)) / x.shape[0]

    @staticmethod
    def MSE(y: ndarray, x: ndarray) -> ndarray:
        """Mean Squared Error function."""
        y = np.atleast_2d(y)
        x = np.atleast_2d(x)
        return np.sum((y - x) ** 2) / x.shape[0]

    @staticmethod
    def dMSE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Mean Squared Error function."""
        y = np.atleast_2d(y)
        x = np.atleast_2d(x)
        ret = -2 * (y - x) / x.shape[0]
        print(x, y, sep="\n\n---\n\n")
        print(ret, end="\n\n")
        return ret

    @staticmethod
    def bias(x: ndarray) -> ndarray:
        """Returns 1."""
        return np.ones(np.atleast_1d(x).shape)

    @staticmethod
    def dbias(x: ndarray) -> ndarray:
        """Returns 0."""
        return np.zeros(np.atleast_1d(x).shape)


def main() -> None:
    """Displays neuron output from dataset input."""
    try:
        print("Neuron presentation:")
        end = False
        while not end:
            try:
                neuron = Neuron(input("\tfunct: "), input("\tderiv: "))
                value = np.random.rand(1)
                print(f"\t\tfunc({value}) = {neuron.eval(value)}")
                print(f"\t\tfunc.deriv({value}) = {neuron.diff(value)}")
                print(neuron)
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
