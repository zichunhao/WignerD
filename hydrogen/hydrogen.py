import scipy.integrate as integrate
import numpy as np
import warnings
import math

warnings.filterwarnings("ignore")
from .wave_func import psi


def hydrogen_matrix_element(
    n1: int,
    l1: int,
    m1: int,
    n2: int,
    l2: int,
    m2: int,
    op=lambda r, theta, phi: 1,
    a: float = 1,
) -> float:
    """
    Matrix element of an operator <n1,l1|op|n2,l2>
    by integrating using the radial wave function.

    :param n1: principal quantum number of the bra.
    :param l1: orbital quantum number of the bra.
    :param m1: magnetic quantum number of the bra.
    :param n2: principal quantum number of the ket.
    :param l2: orbital quantum number of the ket.
    :param m2: magnetic quantum number of the ket.
    :param op: operator (a function of `r`, `theta`, and `phi`).
    :param a: Bohr radius.

    :return: the matrix element.
    """

    def integrand(r, theta, phi):
        bra = np.conj(psi(n1, l1, m1, r, theta, phi, a=a))
        ket = psi(n2, l2, m2, r, theta, phi, a=a)
        return bra * ket * op(r, theta, phi) * (r**2) * np.sin(theta)

    res, _ = integrate.nquad(
        lambda r, theta, phi: integrand(r, theta, phi),
        [[0, math.inf], [0, math.pi], [0, 2 * math.pi]],
    )

    return res
