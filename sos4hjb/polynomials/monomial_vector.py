import numpy as np
from copy import deepcopy

from sos4hjb.polynomials.basis_vector import BasisVector
from sos4hjb.polynomials.polynomial import Polynomial

class MonomialVector(BasisVector):

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        return np.prod([evaluation_dict[v] ** p for v, p in self])

    def __mul__(self, monomial):
        variables = set(list(self.power_dict) + list(monomial.power_dict))
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
        return f'{variable}^{{{power}}}'
