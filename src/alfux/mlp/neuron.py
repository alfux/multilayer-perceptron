import sys
from typing import Callable

import numpy as np
from numpy import ndarray


class Neuron:
    """Neuron node for a neural network.

    Wraps a real-valued function and its derivative. Minimal validation is
    performed for performance.
    """

    def __init__(
            self: "Neuron", f: Callable | str, df: Callable = None
    ) -> None:
        """Define neuron with activation function and derivative.

        Args:
            f (Callable | str): Single-parameter function or the name of a
                predefined function (attribute of ``Neuron``).
            df (Callable | None): Derivative of ``f``. Ignored if ``f`` is a
                string.
        """
        if isinstance(f, str):
            self.eval: Callable = getattr(Neuron, f)
            self.diff: Callable = getattr(Neuron, 'd' + f)
            self._activation = f
        else:
            self.eval: Callable = f
            self.diff: Callable = df
            self._activation = f.__name__

    @property
    def activation(self: "Neuron") -> str:
        """Get the activation function's name."""
        return self._activation

    def eval(self: "Neuron", *args: list) -> ndarray:
        """Compute the neuron's output.

        Args:
            *args: Passed through to the configured function.

        Returns:
            ndarray: Function output.
        """
        pass

    def diff(self: "Neuron", *args: list) -> ndarray:
        """Compute the derivative at the given point.

        Args:
            *args: Passed through to the configured derivative function.

        Returns:
            ndarray: Derivative output.
        """
        pass

    @staticmethod
    def identity(x: ndarray) -> ndarray:
        """Identity function."""
        return x

    @staticmethod
    def didentity(x: ndarray) -> ndarray:
        """Derivative of the identity function."""
        return np.ones(x.shape)

    @staticmethod
    def ReLU(x: ndarray) -> ndarray:
        """Computes the value of Rectified Linear Unit function in x."""
        return np.where(x > 0, x, 0)

    @staticmethod
    def dReLU(x: ndarray) -> ndarray:
        """Computes the differential of ReLU in <x>."""
        return np.where(x > 0, 1, 0)

    @staticmethod
    def LReLU(x: ndarray, a: float = 1e-2) -> ndarray:
        """Computes the value of Leaky Rectified Linear Unit function in x."""
        return np.where(x > 0, x, a * x)

    @staticmethod
    def dLReLU(x: ndarray, a: float = 1e-2) -> ndarray:
        """Computes the differential of LReLU in x."""
        return np.where(x > 0, 1, a)

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
        """Compute the value of the softmax function.

        Args:
            x (ndarray): Input vector or batch.

        Returns:
            ndarray: Softmax output normalized along the last axis.
        """
        out = np.exp(x)
        if x.ndim > 1:
            return out / np.atleast_2d(np.sum(out, axis=1)).T
        return out / np.sum(out)

    @staticmethod
    def dsoftmax(x: ndarray) -> ndarray:
        """Compute the Jacobian of softmax at ``x``."""
        vector = np.exp(x[0])
        vector /= np.sum(vector)
        Jacobian = -np.outer(vector, vector)
        index = np.arange(len(vector))
        Jacobian[index, index] += vector
        return Jacobian

    @staticmethod
    def CELF(y: ndarray, x: ndarray) -> ndarray:
        """Cross-entropy loss function."""
        return -np.sum(
            np.log(np.einsum("ij,ij->i", y, x)), keepdims=True, axis=0
        ) / x.shape[0]

    @staticmethod
    def dCELF(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the cross-entropy loss function.

        This function isn't generalized correctly and only works for one line
        of x, meaning x.shape[0] = 1
        """
        return -np.sum(y / (x + 1e-15), keepdims=True, axis=0) / x.shape[0]

    @staticmethod
    def MSE(y: ndarray, x: ndarray) -> ndarray:
        """Mean squared error (MSE)."""
        return np.sum(
            np.linalg.norm(y - x, axis=1) ** 2,
            keepdims=True
        ) / x.shape[0]

    @staticmethod
    def dMSE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the mean squared error (MSE)."""
        return -2 * (y - x) / x.shape[0]

    @staticmethod
    def MAE(y: ndarray, x: ndarray) -> ndarray:
        """Mean absolute error (MAE)."""
        return np.sum(np.abs(y - x), keepdims=True) / x.shape[0]

    @staticmethod
    def dMAE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the mean absolute error (MAE)."""
        return np.sign(x - y) / x.shape[0]

    @staticmethod
    def MAE_STRONG_0(y: ndarray, x: ndarray) -> ndarray:
        """MAE with stronger penalty when the target is 0."""
        return np.sum(
            np.where(y == 0, 2 * np.abs(y - x), np.abs(y - x)), keepdims=True
        ) / x.shape[0]

    @staticmethod
    def dMAE_STRONG_0(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the MAE_STRONG_0 loss."""
        return np.sum(
            np.where(y == 0, 2 * np.sign(x - y), np.sign(x - y)),
            keepdims=True
        ) / x.shape[0]

    @staticmethod
    def bias(x: ndarray) -> ndarray:
        """Return an array of ones shaped like ``x``."""
        return np.ones(np.atleast_1d(x).shape)

    @staticmethod
    def dbias(x: ndarray) -> ndarray:
        """Return an array of zeros shaped like ``x``."""
        return np.zeros(np.atleast_1d(x).shape)


def main() -> None:
    """Interactively display neuron outputs for sample inputs.

    Returns:
        int: Exit code (``0`` on success, ``1`` on failure).
    """
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
