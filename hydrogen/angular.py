from typing import Callable
import scipy.integrate as integrate
import numpy as np
import math
import warnings

warnings.filterwarnings("ignore")
from .wave_func import Y


def angular_matrix_element(
    l1: int, m1: int, l2: int, m2: int, op: Callable = lambda theta, phi: 1
) -> float:
    """
    Matrix element of an operator (written in the coordinate of theta and phi) <l1,m1|op(theta, phi)|l2,m2>
    by integrating using the spherical harmonics.

    :param l1: orbital quantum number of the bra
    :param m1: magnetic quantum number of the bra
    :param l2: orbital quantum number of the ket
    :param m2: magnetic quantum number of the ket
    :param op: the operator (a function of `theta` and `phi`)

    :return: the matrix element
    """

    def integrand(theta, phi):
        bra = np.conj(Y(l1, m1, theta, phi))
        ket = Y(l2, m2, theta, phi)
        return bra * op(theta, phi) * ket * np.sin(theta)

    res, _ = integrate.nquad(
        lambda theta, phi: integrand(theta, phi), [[0, math.pi], [0, 2 * math.pi]]
    )

    return res
