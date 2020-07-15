from copy import deepcopy
from numpy.polynomial.chebyshev import poly2cheb

import sos4hjb.polynomials as poly

class MonomialVector(poly.BasisVector):
    '''
    Monomial of the form v1 ** p1 * v2 ** p2 * ... * vn ** pn.
    '''

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __mul__(self, monomial):
        self._verify_multiplicand(monomial)
        variables = set(self.variables() + monomial.variables())
        monomial = MonomialVector({v: self[v] + monomial[v] for v in variables})
        return poly.Polynomial({monomial: 1})

    def derivative(self, variable):
        power = self[variable]
        if power == 0:
            monomial = MonomialVector({})
        else:
            monomial = deepcopy(self)
            monomial[variable] -= 1
        return poly.Polynomial({monomial: power})

    def integral(self, variable):
        monomial = deepcopy(self)
        monomial[variable] += 1
        return poly.Polynomial({monomial: 1 / (self[variable] + 1)})

    def in_chebyshev_basis(self):
        res = poly.Polynomial({poly.ChebyshevVector({}): 1})
        for v, p in self:
            c = poly2cheb([0] * p + [1])
            basis = poly.ChebyshevVector.construct_basis([v], p)
            res *= poly.Polynomial(dict(zip(basis, c)))
        return res

    @staticmethod
    def _call_univariate(power, variable):
        return variable ** power

    @staticmethod
    def _repr(variable, power):
        assert power != 0
        if power == 1:
            return f'{variable}'
        else:
            return f'{variable}^{{{power}}}'
