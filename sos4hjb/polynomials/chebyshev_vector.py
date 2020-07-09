import numpy as np
from copy import deepcopy
from itertools import product
from collections import deque

from sos4hjb.polynomials.basis_vector import BasisVector
from sos4hjb.polynomials.polynomial import Polynomial

class ChebyshevVector(BasisVector):
    '''
    Multivariate Chebyshev polynomials of the first kind of the form
    T_p1(v1) * T_p2(v2) * ... * T_pn(vn).
    '''

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        return np.prod([self._call_univariate(p, evaluation_dict[v]) for v, p in self])

    def __mul__(self, cheb):
        self._raise_if_multiplied_by_different_type(cheb)
        variables = set(self.variables + cheb.variables)
        prod_powers = ((self[v] + cheb[v], abs(self[v] - cheb[v])) for v in variables)
        coef = .5 ** len(variables)
        multiplication = Polynomial({})
        for powers in product(*prod_powers):
            cheb = ChebyshevVector(dict(zip(variables, powers)))
            multiplication += Polynomial({cheb: coef})
        return multiplication

    def derivative(self, variable):
        '''
        Uses the formula for the derivative in terms of the polynomials of the
        second kind, then express the polynomial of the second kind as a sum of
        polynomials of the first.
        '''
        power = self[variable]
        derivative = Polynomial({})
        for q in range(power):
            if power % 2 ^ q % 2:
                cheb = deepcopy(self)
                cheb[variable] = q
                derivative[cheb] = power if q == 0 else power * 2
        return derivative

    def primitive(self, variable):
        power = self[variable]
        primitive = Polynomial({})
        cheb = deepcopy(self)
        cheb[variable] = power + 1
        primitive[cheb] = .5 / (1 + power)
        cheb = deepcopy(self)
        cheb[variable] = abs(power - 1)
        primitive[cheb] += .25 if power == 1 else .5 / (1 - power)
        return primitive

    @staticmethod
    def _repr(variable, power):
        return f'T_{{{power}}}({variable})'

    @staticmethod
    def _call_univariate(power, variable):
        T = deque([2 * variable ** 2 - 1, variable], maxlen=2)
        for i in range(power + 1):
            T.append(2 * variable * T[1] - T[0])
        return T[1]
