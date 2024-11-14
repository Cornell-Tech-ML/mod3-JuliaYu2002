"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """Multiplies 2 given numbers together.

    Args:
        x (float): one user given number
        y (float): second user given number

    Return:
        float: the resulting multiplication of the two given numbers
    """
    return x * y


def id(x: float) -> float:
    """Returns the given input.

    Args:
        x (float): a user given number

    Return:
        float: the user's input unmodified
    """
    return x


def add(x: float, y: float) -> float:
    """Adds 2 given numbers together.

    Args:
        x (float): the first given number
        y (float): the second given number

    Return:
        float: the result of the addition
    """
    return x + y


def neg(x: float) -> float:
    """Negates a given number.

    Args:
        x (float): a given number

    Return:
        float: the given number negated
    """
    return -x


def lt(x: float, y: float) -> float:
    """Checks if the first number is smaller than the second.

    Args:
        x (float): the first given number
        y (float): the second given number

    Return:
        float: 1 if the first argument is less than the second, 0 otherwise
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if 2 given numbers are equal.

    Args:
        x (float): first given number
        y (float): second given number

    Return:
        float: 1 if both arguments are equal, 0 otherwise
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Finds the larger number.

    Args:
        x (float): first given number
        y (float): second given number

    Return:
        float: the larger number of the two arguments
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if 2 numbers are close.

    Args:
        x (float): first given number
        y (float): second given number

    Return:
        float: 1 if the two arguments fulfill |x - y| < 1e-2, 0 otherwise
    """
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Calculates the sigmoid of a number.

    Args:
        x (float): user given number

    Return:
        float: the sigmoid of the argument
    """
    # \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Calculates the ReLU of a number.

    Args:
        x (float): user given number

    Return:
        float: the ReLU value of the given number
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Calculates the natural log of a number.

    Args:
        x (float): a user given number

    Return:
        float: the log of the given number
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Calculates the exponential function.

    Args:
        x (float): power of E to calculate

    Return:
        float: result of the exponential E**x
    """
    return math.exp(x)


def inv(x: float) -> float:
    """Gets the inverse of an input.

    Args:
        x (float): user given number

    Return:
        float: the inverse of the input
    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Takes the derivative of log and multiplies it by another number.

    Args:
        x (float): the number that is used in the log derivative
        y (float): the number to multiply by the log derivative

    Return:
        float: the resulting math of d(log(x))/dx * y
    """
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Takes the derivative of the inverse and multiplies it by another number.

    Args:
        x (float): the number that is used in the inverse derivative
        y (float): the number to multiply by the derivative

    Return:
        float: the resulting math of d(x^-1)/dx * y
    """
    return -(1.0 / x**2) * y


def relu_back(x: float, y: float) -> float:
    """Takes the derivative of ReLU and multiplies it by another number.

    Args:
        x (float): the number that is used in the ReLU derivative
        y (float): the number to multiply by the derivative

    Return:
        float: the resulting math of d(relu(x))/dx * y
    """
    return y if x > 0 else 0.0


# ## Task 0.3


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element of a iterable

    Args:
        func (Callable): a user identified function which takes in a float and returns a float

    Return:
        Callable: a usable function that uses an Iterable and returns an Iterable
    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines elements from two iterables using a given function.

    Args:
        func (Callable): a function that takes in 2 floats and returns a float

    Return:
        Callable: takes in 2 Iterables and returns one Iterable combining the elements of the 2 parameters
    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(func(x, y))
        return ret

    return _zipWith


def reduce(
    func: Callable[[float, float], float], x: float
) -> Callable[[Iterable[float]], float]:
    """Reduces elements from an iterable into a single value with a given function.

    Args:
        func (Callable): user given function which takes two floats and returns one
        x (float): a float to track the function operation

    Return:
        Callable: a function which takes in an Iterable and returns a float
    """

    def _reduce(ls: Iterable[float]) -> float:
        val = x
        for l in ls:
            val = func(val, l)
        return val

    return _reduce


def negList(x: Iterable[float]) -> Iterable[float]:
    """Negates all elements in a given iterable by using map and neg.

    Args:
        x (Iterable[float]): an iterable of floats to negate

    Return:
        Iterable[float]: a float iterable containing the negations of the original
    """
    return map(neg)(x)


def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two given iterables.

    Args:
        x (Iterable[float]): one iterable of floats
        y (Iterable[float]): the other iterable of floats

    Return:
        Iterable[float]: an iterable of floats summed according to index value
    """
    return zipWith(add)(x, y)


def sum(x: Iterable[float]) -> float:
    """Adds all elements in an iterable together.

    Args:
        x (Iterable[float]): an iterable of floats

    Return:
        float: the sum of all floats in the given iterable
    """
    return reduce(add, 0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Calculates the product of all elements in a iterable.

    Args:
        x (Iterable[float]): an iterable of floats to multiply together

    Return:
        float: the product of all floats in the given iterable
    """
    return reduce(mul, 1.0)(x)
