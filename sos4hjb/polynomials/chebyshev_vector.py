import numpy as np
from copy import deepcopy
from itertools import product
from collections import deque

from sos4hjb.polynomials.basis_vector import BasisVector
from sos4hjb.polynomials.polynomial import Polynomial

class ChebyshevVector(BasisVector):
    '''
    Product of Chebyshev polynomials of the first kind of the form
    T_p1(v1) * T_p2(v2) * ... * T_pn(vn).

    Attributes
    ----------
    power_dict : dict (key : Variable, value : int)
        Dictionary that maps each variable to its power.
    '''

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        '''
        Parameters
        ----------
        evaluation_dict : dict (key : Variable, value : float)
            Dictionary that assigns a value to every variable of the monomial.
        '''
        return np.prod([self._evaluate_univariate(self[v], value) for v, value in evaluation_dict.items()])

    def __mul__(self, chebyshev):

        # Variables of the product.
        vs = set(list(self.power_dict) + list(chebyshev.power_dict))
        vs_0 = [v for v in vs if self[v] == 0 or chebyshev[v] == 0]

        # Generators for the coefficients of the product.
        product_coef = ((1,) if v in vs_0 else (.5, .5) for v in vs)

        # Generators for the terms of the product.
        add = lambda v: self[v] + chebyshev[v]
        diff = lambda v: abs(self[v] - chebyshev[v])
        product_pow = ((add(v),) if v in vs_0 else (add(v), diff(v)) for v in vs)

        # Expand product.
        coef_dict = {}
        for ps, cs in zip(product(*product_pow), product(*product_coef)):
            chebyshev = ChebyshevVector({v: ps[i] for i, v in enumerate(vs)})
            coef_dict[chebyshev] = np.prod(cs)

        return Polynomial(coef_dict)

    def derivative(self, v):

        # Derivative of 1 is 0.
        p = self[v]
        if p == 0:
            return Polynomial([0], ChebyshevVector({}))

        # Use this as baseline for the ones to be added to the polynomial.
        chebyshev_0 = deepcopy(self)
        chebyshev_0[v] = 0

        # Use the formula for the derivative in terms of the polynomials of the
        # second kind. Then express the polynomial of the second kind as a
        # summation of polynomials of the first.
        coef_dict = {}
        if p % 2:
            coef_dict[deepcopy(chebyshev_0)] = p
        for q in range(1, p):
            if (p % 2 and q % 2 == 0) or (p % 2 == 0 and q % 2):
                chebyshev = deepcopy(chebyshev_0)
                chebyshev[v] = q
                coef_dict[chebyshev] = p * 2

        return Polynomial(coef_dict)

    # ToDo: write more nicely, test, and document.
    def primitive(self, v):
        p = self[v]
        chebyshev_0 = deepcopy(self)
        chebyshev_0[v] = 0
        if p == 0:
            chebyshev_0[v] = 1
            coef_dict = {chebyshev_0: 1}
        elif p == 1:
            chebyshev = deepcopy(chebyshev_0)
            chebyshev[v] = 2
            coef_dict = {chebyshev_0: .25, chebyshev: .25}
        else:
            chebyshev_1 = deepcopy(chebyshev_0)
            chebyshev_1[v] = p + 1
            chebyshev_0[v] = p - 1
            coef_0 = .5 / (1 - p)
            coef_1 = .5 * (p - 1) / (p ** 2 - 1)
            coef_dict = {chebyshev_0: coef_0, chebyshev_1: coef_1}
        return Polynomial(coef_dict)

    @staticmethod
    def _repr(variable, power):
        return f'T_{{{power}}}({variable}) '

    @staticmethod
    def _evaluate_univariate(p, v):
        T = deque([2 * v ** 2 - 1, v], maxlen=2)
        for i in range(p + 1):
            T.append(2 * v * T[1] - T[0])
        return T[1]
