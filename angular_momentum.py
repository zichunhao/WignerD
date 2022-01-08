import numpy as np
import math


def J_x(j):
    return (J_plus(j) + J_minus(j)) / 2

def J_y(j):
    return (J_plus(j) - J_minus(j)) / 2j

def J_z(j):
    return np.identity(int(2 * j + 1))


def Jx(j):
    return J_x(j)

def Jy(j):
    return J_y(j)

def Jz(j):
    return J_z(j)


def J_plus(j):
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f'j must be a half integer. Found: {j}')
    
    dim = int(2 * j + 1)

    mat = np.zeros((dim, dim))

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_plus_component(j, m_prime, j, m)

    return mat

def J_minus(j):
    if int(2 * j + 1) != 2 * j + 1:
        raise ValueError(f'j must be a half integer. Found: {j}')
    
    dim = int(2 * j + 1)

    mat = np.zeros((dim, dim))

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] = J_minus_component(j, m_prime, j, m)

    return mat


def J_plus_component(j_prime, m_prime, j, m):
    '''
    Get the matrix element of the raising operator
    .. math::
        \langle j_\prime, m_\prime | J_+ | j, m \rangle
        \sqrt{(j - m) * (j + m + 1)} \delta_{j_\prime, j} \delta_{m_\prime, m + 1}.
    '''
    if (j_prime != j) or (m_prime != m + 1):
        return 0
    return J_plus_coefficient(j, m)

def J_minus_component(j_prime, m_prime, j, m):
    '''
    Get the matrix element of the lowering operator
    .. math::
        \langle j_\prime, m_\prime | J_+ | j, m \rangle
        \sqrt{(j + m) * (j - m + 1)} \delta_{j_\prime, j} \delta_{m_\prime, m - 1}.
    '''
    if (j_prime != j) or (m_prime != m - 1):
        return 0
    return J_minus_coefficient(j, m)


def J_plus_coefficient(j, m):
    '''
    Applies raising operator on the state :math:`|j, m \rangle`
    and returns the coefficient (:math:`\sqrt{(j + m) (j - m + 1)}`).
    '''
    return math.sqrt((j - m) * (j + m + 1))

def J_minus_coefficient(j, m):
    '''
    Applies lowering operator on the state :math:`|j, m \rangle`
    and returns the coefficient (:math:`\sqrt{(j - m) (j + m + 1)}`).
    '''
    return math.sqrt((j + m) * (j - m + 1))
