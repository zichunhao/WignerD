from pathlib import Path
import sys

# Add project directory as root
path = Path(__file__).parent.absolute()
sys.path.insert(0, str(path.parent))

from angular_momentum import J_x as SU2_Jx, J_y as SU2_Jy, J_z as SU2_Jz
import numpy as np


def J_x(m: float, n: float) -> np.ndarray:
    """Rotation operator for the Lorentz group around the x-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Rotation operator for the Lorentz group around the x-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ji(m, n, 0)


def J_y(m: float, n: float) -> np.ndarray:
    """Rotation operator for the Lorentz group around the y-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Rotation operator for the Lorentz group around the y-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ji(m, n, 1)


def J_z(m: float, n: float) -> np.ndarray:
    """Rotation operator for the Lorentz group around the z-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Rotation operator for the Lorentz group around the z-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ji(m, n, 2)


def K_x(m: float, n: float) -> np.ndarray:
    """Boost operator for the Lorentz group along the x-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Boost operator for the Lorentz group around the x-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ki(m, n, 0)


def K_y(m: float, n: float) -> np.ndarray:
    """Boost operator for the Lorentz group along the y-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Boost operator for the Lorentz group around the y-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ki(m, n, 1)


def K_z(m: float, n: float) -> np.ndarray:
    """Boost operator for the Lorentz group along the z-axis
    for the :math:`(m, n) `representation.

    Reference: https://en.wikipedia.org/wiki/Representation_theory_of_the_Lorentz_group

    :param m: Weight :math:`m` of the representation (Half-integer).
    :type m: float
    :param n: Weight :math:`n` of the representation (Half-integer).
    :type n: int
    :return: Boost operator for the Lorentz group around the z-axis.
    :rtype: np.ndarray
    :raises ValueError: If m or n are not half-integers
    """
    __check_half_ints(m, n)
    return __Ki(m, n, 2)


def __Ji(m: float, n: float, i: int) -> np.ndarray:
    """Generates the rotation operator for the Lorentz group around the i-th axis."""
    Js = (SU2_Jx, SU2_Jy, SU2_Jz)
    return np.tensordot(a=Js[i](m), b=np.eye(int(2 * n + 1)), axes=0) + np.tensordot(
        a=np.eye(int(2 * m + 1)), b=Js[i](n), axes=0
    )


def __Ki(m: float, n: float, i: int) -> np.ndarray:
    """Generates the boost operator for the Lorentz group along the i-th axis."""
    Js = (SU2_Jx, SU2_Jy, SU2_Jz)
    return 1.0j * np.tensordot(
        a=Js[i](m), b=np.eye(int(2 * n + 1)), axes=0
    ) - np.tensordot(a=np.eye(int(2 * m + 1)), b=Js[i](n), axes=0)


def __check_half_ints(m: float, n: float) -> None:
    """Check if m and n are half-integers.
    :raises ValueError: If m is not a half-integer.
    :raises ValueError: If n is not a half-integer.
    """
    if (2 * m) % 2 != 0:
        raise ValueError(f"m must be a half-integer, but m={m} is not.")
    elif (2 * n) % 2 != 0:
        raise ValueError(f"n must be a half-integer, but n={n} is not.")
    else:
        pass


__ALL__ = [
    # rotation operators
    J_x,
    J_y,
    J_z,
    # boost operators
    K_x,
    K_y,
    K_z,
]
