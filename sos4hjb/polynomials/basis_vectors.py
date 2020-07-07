import numpy as np
from copy import deepcopy
from itertools import product
from collections import deque

from sos4hjb.polynomials.basis_vector import BasisVector, is_variable
from sos4hjb.polynomials.polynomial import Polynomial

class MonomialVector(BasisVector):

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        lgtm_evaluation_dict(evaluation_dict, self.power_dict)
        return np.prod([value ** self[v] for v, value in evaluation_dict.items()])

    def __mul__(self, monomial):
        if not self.__class__ is type(monomial):
            raise ValueError(f'''Monomials can only be multiplied with
                monomials, got {type(monomial)}.''')
        vs = self._merge_variables_with(monomial)
        monomial = MonomialVector({v: self[v] + monomial[v] for v in vs})
        return Polynomial([1], [monomial])

    def derivative(self, v):
        p = self[v]
        if p == 0:
            monomial = MonomialVector({})
        else:
            monomial = deepcopy(self)
            monomial[v] -= 1
        return Polynomial([p], [monomial])

    def primitive(self, v):
        c = 1 / (self[v] + 1)
        monomial = deepcopy(self)
        monomial[v] += 1
        return Polynomial([c], [monomial])

    @staticmethod
    def _repr(v, p):
        return f'{v}^{{{p}}} '

class ChebyshevVector(BasisVector):

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        lgtm_evaluation_dict(evaluation_dict, self.power_dict)
        return np.prod([self._evaluate_univariate(self[v], value) for v, value in evaluation_dict.items()])

    def __mul__(self, chebyshev):
        if not self.__class__ is type(chebyshev):
            raise ValueError(f'''Chebyshev vectors can only be multiplied with
                Chebyshev vectors. Got {type(chebyshev)}.''')

        # Variables of the product.
        vs = self._merge_variables_with(chebyshev)
        vs_0 = [v for v in vs if self[v] == 0 or chebyshev[v] == 0]

        # Generators for the coefficients of the product.
        product_coef = ((1,) if v in vs_0 else (.5, .5) for v in vs)

        # Generators for the terms of the product.
        add = lambda v: self[v] + chebyshev[v]
        diff = lambda v: abs(self[v] - chebyshev[v])
        product_pow = ((add(v),) if v in vs_0 else (add(v), diff(v)) for v in vs)

        # Expand product.
        coefficients = list(map(np.prod, product(*product_coef)))
        chebyshevs = [ChebyshevVector({v: ps[i] for i, v in enumerate(vs)}) for ps in product(*product_pow)]

        return Polynomial(coefficients, chebyshevs)

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
        coefficients = []
        chebyshevs = []
        if p % 2:
            coefficients.append(p)
            chebyshevs.append(deepcopy(chebyshev_0))
        for q in range(1, p):
            if (p % 2 and q % 2 == 0) or (p % 2 == 0 and q % 2):
                coefficients.append(p * 2)
                chebyshev = deepcopy(chebyshev_0)
                chebyshev[v] = q
                chebyshevs.append(chebyshev)

        return Polynomial(coefficients, chebyshevs)

    # ToDo: write more nicely, test, and document.
    def primitive(self, v):
        p = self[v]
        chebyshev_0 = deepcopy(self)
        chebyshev_0[v] = 0
        if p == 0:
            coefficients = [1]
            chebyshev_0[v] = 1
            chebyshevs = [chebyshev_0]
        elif p == 1:
            coefficients = [.25, .25]
            chebyshev = deepcopy(chebyshev_0)
            chebyshev[v] = 2
            chebyshevs = [chebyshev_0, chebyshev]
        else:
            coefficients = [.5 * (p - 1) / (p ** 2 - 1), .5 / (1 - p)]
            chebyshev_1 = deepcopy(chebyshev_0)
            chebyshev_1[v] = p + 1
            chebyshev_0[v] = p - 1
            chebyshevs = [chebyshev_1, chebyshev_0]
        return Polynomial(coefficients, chebyshevs)

    @staticmethod
    def _repr(variable, power):
        return f'T_{{{power}}}({variable}) '

    @staticmethod
    def _evaluate_univariate(p, v):
        T = deque([2 * v ** 2 - 1, v], maxlen=2)
        for i in range(p + 1):
            T.append(2 * v * T[1] - T[0])
        return T[1]

def lgtm_evaluation_dict(evaluation_dict, power_dict):
    if not isinstance(evaluation_dict, dict):
        raise ValueError(f'''evaluation_dict must be a dictionary, got
            {type(power_dict)}.''')
    for v in evaluation_dict.keys():
        is_variable(v)
    if not all(v in evaluation_dict for v in power_dict):
        raise ValueError('''evaluation_dict must assign a value to
            each variable in the power_dict.''')
