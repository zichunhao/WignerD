import numpy as np
import math
from typing import Dict, Union
from scipy.linalg import block_diag


def wigner_d(j: int, beta: float) -> np.ndarray:
    dim = int(2 * j + 1)
    mat = np.zeros((dim, dim))

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] += wigner_d_component(j, m_prime, m, beta)

    return mat


def wigner_d_block_diag(jmax: int, beta: float) -> np.ndarray:
    n = int(jmax * 2)
    block = block_diag(*[wigner_d(i / 2, beta) for i in range(n)])
    return block


def wigner_D(j: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    dim = int(2 * j + 1)
    mat = np.zeros((dim, dim), dtype="complex_")

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] += wigner_D_component(j, m_prime, m, alpha, beta, gamma)

    return mat


def wigner_D_block_diag(jmax: int, beta: float) -> np.ndarray:
    n = int(jmax * 2)
    block = block_diag(*[wigner_d(i / 2, beta) for i in range(n)])
    return block


def wigner_D_from_d(wigner_d: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Get the Wigner D matrix from the Wigner little d matrix"""
    dim = wigner_d.shape[0]
    mat = np.zeros((dim, dim), dtype="complex_")

    m_range = np.linspace(-int((dim - 1) / 2), int((dim - 1) / 2), dim)

    for row, m_prime in enumerate(m_range):
        for col, m in enumerate(m_range):
            mat[row, col] = (
                np.exp(-1j * (m_prime * alpha + m * gamma)) * wigner_d[row, col]
            )

    return mat


def wigner_d_from_D(wigner_D: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Get the Wigner little d matrix from the Wigner D matrix"""
    dim = wigner_D.shape[0]
    mat = np.zeros((dim, dim), dtype="complex_")

    m_range = np.linspace(-int((dim - 1) / 2), int((dim - 1) / 2), dim)

    for row, m_prime in enumerate(m_range):
        for col, m in enumerate(m_range):
            mat[row, col] = (
                np.exp(1j * (m_prime * alpha + m * gamma)) * wigner_D[row, col]
            )

    return mat


def wigner_d_dict(jmax: int, beta: float, jmin: int = 0) -> Dict[float, np.ndarray]:
    res = dict()
    jmin = max(jmin, 0)

    for j in range(int(2 * jmin), int(2 * jmax + 1)):
        res[j / 2] = wigner_d(j / 2, beta)
    return res


def wigner_D_dict(
    jmax: int, alpha: float, beta: float, gamma: float, jmin: int = 0
) -> Dict[float, np.ndarray]:
    res = dict()
    jmin = max(jmin, 0)

    for j in range(int(2 * jmin), int(2 * jmax + 1)):
        res[j / 2] = wigner_D(j / 2, alpha, beta, gamma)
    return res


def wigner_D_dict_from_d_dict(
    wigner_d_dict: Dict[float, np.ndarray], alpha: float, gamma: float
) -> Dict[float, np.ndarray]:
    """Get a dictionary of wigner D matrices from a dictionary of wigner d matrices"""
    res = dict()
    for j in wigner_d_dict.keys():
        res[j] = wigner_D_from_d(wigner_d_dict[j], alpha, gamma)
    return res


def wigner_d_dict_from_D_dict(
    wigner_D_dict: Dict[float, np.ndarray], alpha: float, gamma: float
) -> Dict[float, np.ndarray]:
    """Get a dictionary of wigner D matrices from a dictionary of wigner d matrices"""
    res = dict()
    for j in wigner_D_dict.keys():
        res[j] = wigner_d_from_D(wigner_D_dict[j], alpha, gamma)
    return res


def wigner_d_component(j: int, m_prime: int, m: int, beta: int) -> float:
    coefficient = math.sqrt(
        math.factorial(j + m_prime)
        * math.factorial(j - m_prime)
        * math.factorial(j + m)
        * math.factorial(j - m)
    )
    smin = max(0, m - m_prime)
    smax = min(j + m, j - m_prime)

    matrix_element = 0

    s_list = np.linspace(smin, smax, int(smax - smin + 1))

    for s in s_list:
        numerator = (-1) ** (m_prime - m + s)
        numerator *= np.cos(beta / 2) ** (2 * j + m - m_prime - 2 * s)
        numerator *= np.sin(beta / 2) ** (m_prime - m + 2 * s)

        denominator = math.factorial(j + m - s)
        denominator *= math.factorial(s)
        denominator *= math.factorial(m_prime - m + s)
        denominator *= math.factorial(j - m_prime - s)

        matrix_element += numerator / denominator

    return coefficient * matrix_element


def wigner_D_component(
    j: int, m_prime: int, m: int, alpha: float, beta: float, gamma: float
) -> Union[float, complex]:
    return np.exp(-1j * (m_prime * alpha + m * gamma)) * wigner_d_component(
        j, m_prime, m, beta
    )
