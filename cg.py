import math
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple, Union, List
from IPython.display import display

class ClebschGordanCoefficients:
    def __init__(self, jmax):
        if int(2 * jmax) != 2 * jmax:
            jmax = int(2 * jmax) / 2
        
        self.__jmax = jmax
        jmin = 1 / 2
        self.__j_list = np.linspace(jmin, jmax, int(2 * (jmax - jmin) + 1))
        self.__cg_dict = {
            (j1, j2): cg_tables_all_m(j1, j2)
            for j1, j2 in itertools.product(self.__j_list, self.__j_list)
        }

    def __getitem__(self, *args) -> Union[pd.DataFrame, Dict[float, pd.DataFrame]]:
        if len(*args) == 2:
            j1, j2 = args[0]
            return self.__cg_dict.get((j1, j2))
        
        elif len(*args) == 3:
            j1, j2, m = args[0]
            tables = self.__cg_dict.get((j1, j2))
            
            if tables is None:
                return None
            
            return tables.get(m)
        
        else:
            raise IndexError(f'Invalid number of indices: {len(*args)}')
    
    def cg_matrix(self, j1, j2, m, return_indices=False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        mat = self.__cg_dict[(j1, j2)][m]
        
        if return_indices:
          return mat.to_numpy(), mat.index.to_numpy(), mat.columns.to_numpy()
        
        return mat.to_numpy()
    
    def cg_matrices_all_m(self, j1, j2) -> Tuple[
        List[np.ndarray], 
        List[Tuple[
            Tuple[float, float], # (j1, j2)
            float # m
        ]]
    ]:
        '''
        Returns the Clebsch-Gordan coefficients for all m for the given j1, j2 in matrix forms
        along with the index labels of the matrix elements.
        '''
        cg_mats = []
        cg_indices = []
        m_list = self.__cg_dict[(j1, j2)].keys()
        for m in m_list:
            mat, row, col = self.cg_matrix(j1, j2, m, return_indices=True)
            cg_mats.append(mat)
            cg_indices.append((row, col))
        return cg_mats, cg_indices
    
    
    def set_jmax(self, jmax):
        '''Updates jmax and appends larger CG coefficients'''
        if jmax <= self.__jmax:
            return
        
        jmax_old = self.__jmax
        j_new_list = np.linspace(jmax_old, jmax, int(2 * (jmax - jmax_old) + 1))[1:]  # exclude jmax_old
        
        new_dict = dict()
        new_dict.update({
            (j1, j2): cg_tables_all_m(j1, j2)
            for j1, j2 in itertools.product(self.__j_list, j_new_list)
        })
        new_dict.update({
            (j1, j2): cg_tables_all_m(j1, j2)
            for j1, j2 in itertools.product(j_new_list, self.__j_list)
        })
        new_dict.update({
            (j1, j2): cg_tables_all_m(j1, j2)
            for j1, j2 in itertools.product(j_new_list, j_new_list)
        })
        
        self.__cg_dict.update(new_dict)
        self.__j_list = np.concatenate((self.__j_list, j_new_list))
        self.__jmax = jmax
        
        return True
        
    def __repr__(self) -> str:
        return f'ClebschGordanCoefficients(jmax={self.__jmax})'
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def display_tables(self, jmin=None, jmax=None):
        '''
        Displays all CG coefficients stored in the object
            If the bounds jmin and/or jmax are/is specified, display within the given bounds 
        '''
        
        for key, tables in self.__cg_dict.items():
            j1, j2 = key
            if (jmax is not None and jmax > 0) and (jmax < j1 or jmax < j2):
                continue
            
            if (jmin is not None and jmin > 0) and (jmin > j1 or jmin > j2):
                continue
            
            print(f'{j1 = }, {j2 = }')
            for m, table in tables.items():
                print(f'{m = }:')
                display(table)
        return
    
    @property
    def dicts(self):
        return self.__cg_dict
    
    @property
    def jmax(self):
        return self.__jmax
    
    @property
    def keys(self):
        return self.__cg_dict.keys()
        
    

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
