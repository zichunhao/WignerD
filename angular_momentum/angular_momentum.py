import numpy as np
from .ladder_ops import J_plus, J_minus


def J_x(j: int) -> np.ndarray:
    return (J_plus(j) + J_minus(j)) / 2


def J_y(j: int) -> np.ndarray:
    return (J_plus(j) - J_minus(j)) / 2j


def J_z(j: int) -> np.ndarray:
    return np.identity(int(2 * j + 1))


def Jx(j: int) -> np.ndarray:
    return J_x(j)


def Jy(j: int) -> np.ndarray:
    return J_y(j)


def Jz(j: int) -> np.ndarray:
    return J_z(j)
