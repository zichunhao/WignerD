import numpy as np
from .ladder_ops import J_plus, J_minus

def J_x(j) -> np.ndarray:
    return (J_plus(j) + J_minus(j)) / 2

def J_y(j) -> np.ndarray:
    return (J_plus(j) - J_minus(j)) / 2j

def J_z(j) -> np.ndarray:
    return np.identity(int(2 * j + 1))


def Jx(j) -> np.ndarray:
    return J_x(j)

def Jy(j) -> np.ndarray:
    return J_y(j)

def Jz(j) -> np.ndarray:
    return J_z(j)

