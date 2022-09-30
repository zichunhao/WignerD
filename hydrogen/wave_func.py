from scipy.special import sph_harm, assoc_laguerre
import math
FACTORIAL_LIMIT = 170  # the limit over which factorial cannot be converted to float

def Y(
    l: int, 
    m: int, 
    theta: float, 
    phi: float
) -> complex:
    '''
    Spherical harmonics in the convention of physics.
    
    :param l: orbital quantum number.
    :param m: magnetic quantum number.
    :param theta: azimuthal angle (angle with z axis).
    :param phi: polar angle (angle with x axis in the xy plane).
    
    :return: the spherical harmonics Y_m^l(theta, phi).
    '''
    
    if l < 0:
        raise ValueError(f'l must be larger than 0. Found: l = {l}')
    if m < -l or m > l:
        raise ValueError(f'm must be between -l and l. Found: m = {m}, l = {l}')
    
    return sph_harm(m, l, phi, theta)


def R(
    n: int, 
    l: int, 
    r: float, 
    a: float = 1
) -> float:
    '''
    Radial wave function of the hydrogen atom :math:R_{n \ell}(r)
    
    :param n: principal quantum number.
    :param l: orbital quantum number.
    :param r: dimensionless radius.
    :param a: bohr radius, default to 1, 
    in which case r is in atomic unit of length.
    
    :return: normalized radial wave function at r.
    '''
    n = int(n)
    
    if n < 1:
        raise ValueError(f'n must be greater than 0. Found: n = {n}')
    if l < 0 or l >= n:
        raise ValueError(
            f'l must be greater than or equal to 0 and less than n. Found: n = {n}, l = {l}.'
        )

    norm = math.sqrt((2 / (n * a)) ** 3)
    
    if n - l - 1 <= FACTORIAL_LIMIT:
        norm *= math.sqrt(math.factorial((n - l - 1)) / (2 * n) / math.factorial(n + l))
    else:
        norm /= math.sqrt((2 * n) * prod(n-l, n+l))
    
    norm *= math.exp(- r / (n * a))
    norm *= (2*r / (n * a)) ** l

    rad_func = assoc_laguerre(x=2*r/(n*a), n=n-l-1, k=2*l+1)

    return norm * rad_func

def psi(
    n: int, l: int, m: int, 
    r: float, theta: float, phi: float,
    a: float = 1
) -> float:
    '''
    Normalized wave function of the hydrogen atom.
    
    :param n: principal quantum number.
    :param l: orbital quantum number.
    :param m: magnetic quantum number.
    :param r: radius.
    :param theta: azimuthal angle [0, pi].
    :param phi: polar angle [0, 2pi].
    :param a: Bohr radius.
    
    :return: normalized wave function psi_{n, l, m} at r, theta, phi.
    '''
    
    n = int(n)
    
    if n < 1:
        raise ValueError(f'n must be greater than 0. Found: n = {n}')
    if l < 0 or l >= n:
        raise ValueError(f'l must be between 0 and n - 1. Found: l = {l}, n = {n}')
    if m < -l or m > l:
        raise ValueError(f'm must be between -l and l. Found: m = {m}, l = {l}')
    
    return R(n, l, r, a) * Y(l, m, theta, phi)


def prod(start: int, end: int) -> int:
    p = 1
    for i in range(start, end+1):
        p *= i
    return p
