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
        variables = set(list(self.power_dict) + list(cheb.power_dict))
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
        if power > 0:
            for q in range(power):
                if (power % 2 and q % 2 == 0) or (power % 2 == 0 and q % 2):
                    cheb = deepcopy(self)
                    cheb[variable] = q
                    derivative[cheb] = power if q == 0 else power * 2
        return derivative

    # ToDo: write more nicely, test, and document.
    def primitive(self, variable):
        power = self[variable]
        cheb_0 = deepcopy(self)
        cheb_0[variable] = 0
        if power == 0:
            cheb_0[variable] = 1
            coef_dict = {cheb_0: 1}
        elif power == 1:
            cheb = deepcopy(cheb_0)
            cheb[variable] = 2
            coef_dict = {cheb_0: .25, cheb: .25}
        else:
            cheb_1 = deepcopy(cheb_0)
            cheb_1[variable] = power + 1
            cheb_0[variable] = power - 1
            coef_0 = .5 / (1 - power)
            coef_1 = .5 * (power - 1) / (power ** 2 - 1)
            coef_dict = {cheb_0: coef_0, cheb_1: coef_1}
        return Polynomial(coef_dict)

    @staticmethod
    def _repr(variable, power):
        return f'T_{{{power}}}({variable})'

    @staticmethod
    def _call_univariate(power, variable):
        T = deque([2 * variable ** 2 - 1, variable], maxlen=2)
        for i in range(power + 1):
            T.append(2 * variable * T[1] - T[0])
        return T[1]
