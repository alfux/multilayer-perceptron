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

    def __init__(self: Self, f: Callable | str, df: Callable = None) -> None:
        """Define neuron with its activation function and derivative.

        Args:
            f must be a single parameter real function or the name of a
            predefined Neuron.
            df must be the derivative of f. If f is a string, df is ignored.
        """
        if isinstance(f, str):
            self.eval: Callable = eval("Neuron." + f)
            self.diff: Callable = eval("Neuron.d" + f)
        else:
            self.eval: Callable = f
            self.diff: Callable = df

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
        """Computes the value of softmax function in x.

        Args:
            x (ndarray): The input vector.
        Returns:
            ndarray: The output value.
        """
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
        return -np.sum(
            np.log(np.einsum("ij,ij->i", y, x)), keepdims=True
        ) / x.shape[0]

    @staticmethod
    def dCELF(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Cross Entropy Loss Function."""
        return -np.sum(y / (x + 1e-15), keepdims=True) / x.shape[0]

    @staticmethod
    def MSE(y: ndarray, x: ndarray) -> ndarray:
        """Mean Squared Error function."""
        return np.sum((y - x) ** 2, keepdims=True) / x.shape[0]

    @staticmethod
    def dMSE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Mean Squared Error function."""
        return -2 * (y - x) / x.shape[0]

    @staticmethod
    def MAE(y: ndarray, x: ndarray) -> ndarray:
        """Mean Absolute Error."""
        return np.sum(np.abs(y - x), keepdims=True) / x.shape[0]

    @staticmethod
    def dMAE(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Mean Absolute Error."""
        return np.sign(x - y) / x.shape[0]

    @staticmethod
    def MAE_STRONG_0(y: ndarray, x: ndarray) -> ndarray:
        """Mean Absolute Error with strong weight on 0's."""
        return np.sum(
            np.where(y == 0, 2 * np.abs(y - x), np.abs(y - x)), keepdims=True
        ) / x.shape[0]

    @staticmethod
    def dMAE_STRONG_0(y: ndarray, x: ndarray) -> ndarray:
        """Derivative of the Mean Absolute Error STRONG 0."""
        return np.sum(
            np.where(y == 0, 2 * np.sign(x - y), np.sign(x - y)),
            keepdims=True
        ) / x.shape[0]

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
