import numpy as np
from typing import Tuple, Dict
from scipy.linalg import block_diag
from .methods import *


class WignerD:
    def __init__(self, jmax, alpha, beta, gamma):
        """
        An object for the Wigeon D matrix
        given the Euler angles (`alpha`, `beta`, `gamma`)
        and the maximum j (`jmax`).
        """
        self.__jmax = jmax
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__d = wigner_d_dict(jmax, beta)
        # self.__D = wigner_D_dict(jmax, alpha, beta, gamma)
        self.__D = wigner_D_dict_from_d_dict(self.__d, alpha, gamma)

    def wigner_D(self, j) -> np.ndarray:
        try:
            return self.__D[j]
        except KeyError:
            self.set_jmax(j)
            return self.__D[j]

    def wigner_d(self, j) -> np.ndarray:
        try:
            return self.__d[j]
        except KeyError:
            self.set_jmax(j)
            return self.__d[j]

    def get_d_block(self, jmax=None) -> np.ndarray:
        """
        Get the block diagonal form of the Wigner d matrix
            - up to `jmax` if specified
            - the stored `jmax` otherwise.
        """
        if jmax is not None:
            return block_diag(*[self.d[j / 2] for j in range(int(2 * jmax + 1))])
        return block_diag(*[self.d[j / 2] for j in range(int(2 * self.jmax + 1))])

    def get_D_block(self, jmax=None) -> np.ndarray:
        """
        Get the block diagonal form of the Wigner D matrix
            - up to `jmax` if specified
            - the stored `jmax` otherwise.
        """
        if jmax is not None:
            return block_diag(*[self.D[j / 2] for j in range(int(2 * jmax + 1))])
        return block_diag(*[self.D[j / 2] for j in range(int(2 * self.jmax + 1))])

    def set_jmax(self, jmax):
        if jmax <= self.__jmax:
            return

        jmin = max(self.__d.keys())
        wigner_d_new = wigner_d_dict(jmax, self.__beta, jmin=jmin)
        self.__d.update(wigner_d_new)
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

    def __getitem__(self, j) -> Tuple[np.ndarray, np.ndarray]:
        return (self.wigner_d(j), self.wigner_D(j))

    def __repr__(self) -> str:
        return f"WignerDDict(jmax={self.__jmax}, alpha={self.__alpha}, beta={self.__beta}, gamma={self.__gamma})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def jmax(self) -> float:
        return self.__jmax

    @property
    def alpha(self) -> float:
        return self.__alpha

    @property
    def beta(self) -> float:
        return self.__beta

    @property
    def gamma(self) -> float:
        return self.__gamma

    @property
    def d(self) -> Dict[float, np.ndarray]:
        return self.__d

    @property
    def D(self) -> Dict[float, np.ndarray]:
        return self.__D

    @property
    def d_block(self):
        return self.get_d_block()

    @property
    def D_block(self):
        return self.get_D_block()
