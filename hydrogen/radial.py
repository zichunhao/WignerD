import scipy.integrate as integrate
import math
import warnings
warnings.filterwarnings('ignore')
from .wave_func import R

def radial_matrix_element(
    n1: int, l1: int, n2: int, l2: int, 
    a: float = 1, op = lambda r: 1
) -> float:
    '''
    Matrix element of an operator <n1,l1|op|n2,l2>
    by integrating using the radial wave function.
    
    :param n1: principal quantum number of the bra.
    :param l1: orbital quantum number of the bra.
    :param n2: principal quantum number of the ket.
    :param l2: orbital quantum number of the ket.
    :param op: operator (a function of `r`).
    :param a: Bohr radius.
    
    :return: the matrix element.
    '''
    
    def integrand(r):
        bra = R(n1, l1, r, a)
        ket = R(n2, l2, r, a)
        return bra * ket * op(r) * (r ** 2)
    
    res, _ = integrate.quad(integrand, 0, math.inf)
    
    return res

