"""Module with common mathematical operators that could be reused elsewhere"""


def derivative1_forward_h2(target, y):
    """
    Implements the taylor approach to calculate derivatives, giving the point of interest.
    The function used is y=f(target), where len(y)-2 > target >= 0
    :param target: the position to be derived
    :param y: an array of points with the values
    :return: the derivative of the target with the given values
    """
    if len(y) - 3 < target or target < 0:
        raise(ValueError("need two more points to calculate the derivative"))
    return (-y[target+2] + 4*y[target+1] - 3*y[target])/2


def derivative1_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the first derivative.

    :param target: the position to be derived, must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise(ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - y[target - 1])/2


def derivative2_centered_h1(target, y):
    """
    Implements the taylor centered approach to calculate the second derivative.

    :param target: the position to be derived,  must be len(y)-1 > target > 0
    :param y: an array with the values
    :return: the centered second derivative of target
    """
    if len(y) - 1 <= target <= 0:
        raise(ValueError("Invalid target, array size {}, given {}".format(len(y), target)))
    return (y[target + 1] - 2*y[target] + y[target - 1])/4
