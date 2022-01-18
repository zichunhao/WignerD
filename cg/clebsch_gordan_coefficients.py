import itertools
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List
from .methods import *

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
            if j1 > self.jmax or j2 > self.jmax:
                self.set_jmax(max(j1, j2))
            return self.__cg_dict.get((j1, j2))
        
        elif len(*args) == 3:
            j1, j2, m = args[0]
            if j1 > self.jmax or j2 > self.jmax:
                self.set_jmax(max(j1, j2))
            return self.__cg_dict.get((j1, j2)).get(m)
        
        else:
            raise IndexError(f'Invalid number of indices: {len(*args)}')
    
    def cg_matrix(self, j1, j2, m, return_indices=False) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        cg_dict = self.__cg_dict.get((j1, j2))
        if cg_dict is None:
            self.update(max(j1, j2))
        
        mat = cg_dict.get(m)
        
        if mat is None:
            return None
        
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
        
        if j1 > self.jmax or j2 > self.jmax:
            self.set_jmax(max(j1, j2))
        
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
    
    def display_all_tables(self, jmin=None, jmax=None):
        '''
        Displays all CG coefficients stored in the object
            If the bounds jmin and/or jmax are/is specified, display within the given bounds 
        '''
        if jmax is not None and jmax > self.jmax:
            self.set_jmax(jmax)
        
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
    def dicts(self) -> Dict[
        Tuple[float, float],
        Dict[float, pd.DataFrame]
    ]:
        return self.__cg_dict
    
    @property
    def jmax(self):
        return self.__jmax
    
    @property
    def keys(self):
        return self.__cg_dict.keys()
