import numpy as np
from copy import deepcopy

from sos4hjb.polynomials.basis_vector import BasisVector
from sos4hjb.polynomials.polynomial import Polynomial

class MonomialVector(BasisVector):
    '''
    Monomial of the form v1 ** p1 * v2 ** p2 * ... * vn ** pn.
    '''

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        return np.prod([evaluation_dict[v] ** p for v, p in self])

    def __mul__(self, monomial):
        self._raise_if_multiplied_by_different_type(monomial)
        variables = set(self.variables + monomial.variables)
        monomial = MonomialVector({v: self[v] + monomial[v] for v in variables})
        return Polynomial({monomial: 1})

    def derivative(self, variable):
        power = self[variable]
        if power == 0:
            monomial = MonomialVector({})
        else:
            monomial = deepcopy(self)
            monomial[variable] -= 1
        return Polynomial({monomial: power})

    def primitive(self, variable):
        monomial = deepcopy(self)
        monomial[variable] += 1
        return Polynomial({monomial: 1 / (self[variable] + 1)})

    @staticmethod
    def _repr(variable, power):
        assert power != 0
        if power == 1:
            return f'{variable}'
        else:
            return f'{variable}^{{{power}}}'
