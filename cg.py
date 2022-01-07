import math
import numpy as np

def cg_coefficient(
    j1, j2, 
    m1, m2,
    j, m
):
    '''
    Compute the Clebsch-Gordan coefficient :math:`\langle j_1, j_2; m_1, m_2 | j_1 j_2; j m \rangle`.
    Reference: Section 3.8 of J. J. Sakurai, "Quantum Mechanics", 2nd ed., Springer, 2008
    '''
    
    if m != m1 + m2:
        return 0
    if j < abs(j1 - j2):
        return 0
    if j > j1 + j2:
        return 0
    
    # coefficient outside of the summation
    numerator = 2 * j + 1
    numerator *= math.factorial(j + j1 - j2)
    numerator *= math.factorial(j - j1 + j2)
    numerator *= math.factorial(j1 + j2 - j)
    numerator *= math.factorial(j + m)
    numerator *= math.factorial(j - m)
    numerator *= math.factorial(j1 - m1)
    numerator *= math.factorial(j1 + m1)
    numerator *= math.factorial(j2 - m2)
    numerator *= math.factorial(j2 + m2)
    
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
    