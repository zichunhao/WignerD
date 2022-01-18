import math
import itertools
import numpy as np
import pandas as pd
from typing import Dict
from IPython.display import display

def cg_coefficient(
    j1, j2,
    m1, m2,
    j, m
):
    '''
    Compute the Clebsch-Gordan coefficient :math:`\langle j_1, j_2; m_1, m_2 | j_1, j_2; j, m \rangle`.
    References: 
        - Section 3.8 of J. J. Sakurai and Jim Napolitano, "Quantum Mechanics", 2nd ed., Cambridge, 2017
        - https://en.wikipedia.org/wiki/Table_of_Clebsch–Gordan_coefficients
    '''

    if m != m1 + m2:
        return 0
    if j < abs(j1 - j2):
        return 0
    if j > j1 + j2:
        return 0

    # coefficient outside of the summation
    numerator = 2 * j + 1
    try:
        numerator *= math.factorial(j + j1 - j2)
        numerator *= math.factorial(j - j1 + j2)
        numerator *= math.factorial(j1 + j2 - j)
        numerator *= math.factorial(j + m)
        numerator *= math.factorial(j - m)
        numerator *= math.factorial(j1 - m1)
        numerator *= math.factorial(j1 + m1)
        numerator *= math.factorial(j2 - m2)
        numerator *= math.factorial(j2 + m2)
    except ValueError:  # invalid j1, j2, m1, m2, j, or m -> 0
        return 0

    denominator = math.factorial(j1 + j2 + j + 1)

    c = math.sqrt(numerator / denominator)

    # summation
    summation = 0
    # ranges of k
    k_min = max(0, -(j - j2 + m1), -(j - j1 - m2))
    k_max = min(j1 + j2 - j, j1 - m1, j2 + m2)
    k_max = max(0, k_max)  # avoid negative k's
    k_list = np.linspace(k_min, k_max, int(k_max - k_min + 1))

    for k in k_list:
        numerator = (-1) ** k

        denominator = math.factorial(k)
        denominator *= math.factorial(j1 + j2 - j - k)
        denominator *= math.factorial(j1 - m1 - k)
        denominator *= math.factorial(j2 + m2 - k)
        denominator *= math.factorial(j - j2 + m1 + k)
        denominator *= math.factorial(j - j1 - m2 + k)

        summation += numerator / denominator

    return c * summation


def cg_table(j1, j2, m) -> pd.DataFrame:
    '''
    Returns the Clebsch-Gordan table given :math:`j_1, j_2, m`.
    '''
    table = dict()
    m1_list = np.linspace(-j1, j1, int(2*j1 + 1))
    m2_list = np.linspace(-j2, j2, int(2*j2 + 1))
    j_list = list(set(
        abs(m1 + m2)
        for m1, m2 in list(itertools.product(m1_list, m2_list))
        if abs(m1 + m2) >= abs(m)  # constraints
    ))

    for m1, m2 in list(itertools.product(m1_list, m2_list)):
        if m1 + m2 != m:
            continue
        key = (m1, m2)
        value = []
        for j in j_list:
            coefficient = cg_coefficient(
                j1=j1, j2=j2,
                m=m1+m2,
                j=j,
                m1=m1, m2=m2,
            )
            value.append(coefficient)
        table[key] = value

    df = pd.DataFrame.from_dict(table, orient='index', columns=j_list)
    df.index.name = '(m1, m2)'
    df.columns.name = 'j'
    return df


def cg_matrix(j1, j2, m, return_indices=False):
    '''
    Returns the Clebsch-Gordan matrix given :math:`j_1, j_2, m` in matrix form.
    Returns row indices (`m1, m2`) and column indices (`j`) if `return_indices` is `True`.
    '''
    table = cg_table(j1, j2, m)

    if return_indices:
        return table.to_numpy(), table.index.to_numpy(), table.columns.to_numpy()

    return table.to_numpy()


def cg_tables_all_m(j1, j2, display_tables=False) -> Dict:
    '''
    Returns a dictionary of CClebsch-Gordan table given :math:`j_1, j_2`
    with all possible :math:`m`.
    '''
    tables = dict()
    m_max = j1 + j2
    m_min = - m_max
    m_list = np.linspace(m_min, m_max, int((m_max - m_min) + 1))

    for m in m_list:
        tables[m] = cg_table(j1, j2, m)

    if display_tables:
        print(f'{j1 = }, {j2 = }')
        for m, table in tables.items():
            print(f'{m = }:')
            display(table)

    return tables
