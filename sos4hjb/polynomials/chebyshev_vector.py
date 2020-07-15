from math import cos, acos, cosh, acosh
from copy import deepcopy
from itertools import product
from numpy.polynomial.chebyshev import cheb2poly

import sos4hjb.polynomials as poly

class ChebyshevVector(poly.BasisVector):
    '''
    Multivariate Chebyshev polynomials of the first kind of the form
    T_p1(v1) * T_p2(v2) * ... * T_pn(vn).
    '''

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __mul__(self, cheb):
        self._verify_multiplicand(cheb)
        variables = set(self.variables() + cheb.variables())
        prod_powers = ((self[v] + cheb[v], abs(self[v] - cheb[v])) for v in variables)
        coef = .5 ** len(variables)
        multiplication = poly.Polynomial({})
        for powers in product(*prod_powers):
            cheb = ChebyshevVector(dict(zip(variables, powers)))
            multiplication += poly.Polynomial({cheb: coef})
        return multiplication

    def derivative(self, variable):
        '''
        Uses the formula for the derivative in terms of the polynomials of the
        second kind, then express the polynomial of the second kind as a sum of
        polynomials of the first.
        '''
        power = self[variable]
        derivative = poly.Polynomial({})
        for q in range(power):
            if power % 2 ^ q % 2:
                cheb = deepcopy(self)
                cheb[variable] = q
                derivative[cheb] = power if q == 0 else power * 2
        return derivative

    def integral(self, variable):
        power = self[variable]
        integral = poly.Polynomial({})
        cheb = deepcopy(self)
        cheb[variable] = power + 1
        integral[cheb] = .5 / (1 + power)
        cheb = deepcopy(self)
        cheb[variable] = abs(power - 1)
        integral[cheb] += .25 if power == 1 else .5 / (1 - power)
        return integral

    def in_monomial_basis(self):
        res = poly.Polynomial({poly.MonomialVector({}): 1})
        for v, p in self:
            c = cheb2poly([0] * p + [1])
            basis = poly.MonomialVector.construct_basis([v], p)
            res *= poly.Polynomial(dict(zip(basis, c)))
        return res

    @staticmethod
    def _repr(variable, power):
        assert power != 0
        return f'T_{{{power}}}({variable})'

    @staticmethod
    def _call_univariate(power, variable):
        if abs(variable) <= 1:
            return cos(power * acos(variable))
        elif variable > 1:
            return cosh(power * acosh(variable))
        else:
            sign = - 1 if power % 2 else 1
            return sign * cosh(power * acosh(- variable))
