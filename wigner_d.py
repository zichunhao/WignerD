import numpy as np
import math

class WignerD:
    def __init__(self, jmax, alpha, beta, gamma):
        '''
        An object for the Wigeon D matrix 
        given the Euler angles (`alpha`, `beta`, `gamma`) 
        and the maximum j (`jmax`).
        '''
        self.__jmax = jmax
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__d = wigner_d_dict(jmax, beta)
        # self.__D = wigner_D_dict(jmax, alpha, beta, gamma)
        self.__D = wigner_D_dict_from_d_dict(self.__d, alpha, gamma)

    def wigner_D(self, j):
        try:
            return self.__D[j]
        except KeyError:
            self.set_jmax(j)
            return self.__D[j]

    def wigner_d(self, j):
        try:
            return self.__d[j]
        except KeyError:
            self.set_jmax(j)
            return self.__d[j]
    
    def set_jmax(self, jmax):
        if jmax <= self.__jmax:
            return
        
        jmin = max(self.__d.keys())
        wigner_d_new = wigner_d_dict(jmax, self.__beta, jmin=jmin)
        self.__d.update(
            wigner_d_new
        )
        self.__D.update(
            wigner_D_dict_from_d_dict(wigner_d_new, self.__alpha, self.__gamma)
        )
        
        self.__jmax = jmax
        return

    def set_alpha(self, alpha):
        self.__D = wigner_D_dict(self.__jmax, alpha, self.__beta, self.__gamma)
        self.__alpha = alpha
        return
        
    def set_beta(self, beta):
        self.__D = wigner_D_dict(self.__jmax, self.__alpha, beta, self.__gamma)
        self.__d = wigner_d_dict(self.__jmax, beta)
        self.__beta = beta
        return
        
    def set_gamma(self, gamma):
        self.__D = wigner_D_dict(self.__jmax, self.__alpha, self.__beta, gamma)
        self.__gamma = gamma
        return
    
    def __getitem__(self, j):
        return (self.wigner_d(j), self.wigner_D(j))

    def __repr__(self) -> str:
        return f'WignerDDict(jmax={self.__jmax}, alpha={self.__alpha}, beta={self.__beta}, gamma={self.__gamma})'

    def __str__(self) -> str:
        return self.__repr__()
    
    @property
    def jmax(self):
        return self.__jmax
    
    @property
    def alpha(self):
        return self.__alpha
    
    @property
    def beta(self):
        return self.__beta
    
    @property
    def gamma(self):
        return self.__gamma
    
    @property
    def d(self):
        return self.__d
    
    @property
    def D(self):
        return self.__D


def wigner_d(j, beta):
    dim = int(2 * j + 1)
    mat = np.zeros((dim, dim))

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row, col] += wigner_d_component(j, m_prime, m, beta)

    return mat


def wigner_D(j, alpha, beta, gamma):
    dim = int(2 * j + 1)
    mat = np.zeros((dim, dim), dtype='complex_')

    m_prime_list = np.linspace(-j, j, dim)
    m_list = np.linspace(-j, j, dim)

    for row, m_prime in enumerate(m_prime_list):
        for col, m in enumerate(m_list):
            mat[row,col] += wigner_D_component(
                j, m_prime, m,
                alpha, beta, gamma
            )

    return mat


def wigner_D_from_d(wigner_d, alpha, gamma):
    '''Get the Wigner D matrix from the Wigner little d matrix'''
    dim = wigner_d.shape[0]
    mat = np.zeros((dim, dim), dtype='complex_')

    m_range = np.linspace(-int((dim-1) / 2), int((dim-1) / 2), dim)

    for row, m_prime in enumerate(m_range):
        for col, m in enumerate(m_range):
            mat[row,col] = np.exp(-1j * (m_prime * alpha + m * gamma)) * wigner_d[row,col]

    return mat


def wigner_d_from_D(wigner_D, alpha, gamma):
    '''Get the Wigner little d matrix from the Wigner D matrix'''
    dim = wigner_D.shape[0]
    mat = np.zeros((dim, dim), dtype='complex_')

    m_range = np.linspace(-int((dim-1) / 2), int((dim-1) / 2), dim)

    for row, m_prime in enumerate(m_range):
        for col, m in enumerate(m_range):
            mat[row, col] = np.exp(1j * (m_prime *alpha + m * gamma)) * wigner_D[row, col]

    return mat


def wigner_d_dict(jmax, beta, jmin=0):
    res = dict()
    jmin = max(jmin, 0)
    
    for j in range(int(2 * jmin), int(2 * jmax + 1)):
        res[j / 2] = wigner_d(j / 2, beta)
    return res


def wigner_D_dict(jmax, alpha, beta, gamma, jmin=0):
    res = dict()
    jmin = max(jmin, 0)

    for j in range(int(2 * jmin), int(2 * jmax + 1)):
        res[j / 2] = wigner_D(j / 2, alpha, beta, gamma)
    return res


def wigner_D_dict_from_d_dict(wigner_d_dict, alpha, gamma):
    '''Get a dictionary of wigner D matrices from a dictionary of wigner d matrices'''
    res = dict()
    for j in wigner_d_dict.keys():
        res[j] = wigner_D_from_d(wigner_d_dict[j], alpha, gamma)
    return res


def wigner_d_dict_from_D_dict(wigner_D_dict, alpha, gamma):
    '''Get a dictionary of wigner D matrices from a dictionary of wigner d matrices'''
    res = dict()
    for j in wigner_D_dict.keys():
        res[j] = wigner_d_from_D(wigner_D_dict[j], alpha, gamma)
    return res


def wigner_d_component(j, m_prime, m, beta):
    coefficient = math.sqrt(
        math.factorial(j + m_prime) * 
        math.factorial(j - m_prime) *
        math.factorial(j + m) * 
        math.factorial(j - m)
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


def wigner_D_component(j, m_prime, m, alpha, beta, gamma):
    return np.exp(-1j * (m_prime * alpha + m * gamma)) * wigner_d_component(j, m_prime, m, beta)