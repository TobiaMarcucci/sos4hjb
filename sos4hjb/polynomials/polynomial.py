from math import prod
from operator import eq, ne, gt
from numbers import Number
from copy import deepcopy

import sos4hjb.polynomials as poly

class Polynomial:
    '''
    Polynomial expressed as the linear combination of basis vectors. Written in
    such a way that coef_dict never contains an item with value equal to zero.

    Attributes
    ----------
    coef_dict : dict (key : BasisVector, value : float)
        Dictionary that maps each basis vector to its coefficient.
    '''

    def __init__(self, coef_dict):
        self._verify_vectors(coef_dict.keys())
        self.coef_dict = {v: c for v, c in coef_dict.items() if optimistic(c, ne, 0)}

    def __getitem__(self, vector):
        return self.coef_dict[vector] if vector in self.coef_dict else 0

    def __setitem__(self, vector, coef):
        if pessimistic(coef, eq, 0):
            self.coef_dict.pop(vector, None)
        else:
            self.coef_dict[vector] = coef
            self._verify_vectors(self.vectors())

    def __eq__(self, other):
        other = Polynomial({}) if other == 0 else other
        return self.coef_dict == other.coef_dict
    
    def __ne__(self, other):
        return not self == other

    def __len__(self):
        return len(self.coef_dict)

    def __iter__(self):
        return iter(self.coef_dict.items())

    def __call__(self, evaluation_dict):
        return sum(v(evaluation_dict) * c for v, c in self)

    def substitute(self, evaluation_dict):
        return sum(v.substitute(evaluation_dict) * c for v, c in self)

    def __pos__(self):
        return deepcopy(self)

    def __neg__(self):
        return Polynomial({v: - c for v, c in self})

    def __abs__(self):
        return Polynomial({v: abs(c) for v, c in self})

    def __round__(self, digits=0):
        return Polynomial({v: round(c, digits) for v, c in self})

    def __add__(self, other):
        if isinstance(other, Polynomial):
            vectors = set(self.vectors() + other.vectors())
            return Polynomial({v: self[v] + other[v] for v in vectors})
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Polynomial):
            for v, c in other:
                self[v] += c
            return self
        else:
            return NotImplemented

    def __radd__(self, other):
        # Defines 0 + self. Useful to use sum() on a list of polynomials.
        if pessimistic(other, eq, 0):
            return deepcopy(self)
        else:
            return NotImplemented

    def __sub__(self, other):
        # Does not use __add__ to avoid the overhead of __neg__.
        if isinstance(other, Polynomial):
            vectors = set(self.vectors() + other.vectors())
            return Polynomial({v: self[v] - other[v] for v in vectors})
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, Polynomial):
            for v, c in other:
                self[v] -= c
            return self
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            return sum((vs * vo) * (cs * co) for vs, cs in self for vo, co in other)
        else: # Tries to treat other as a scalar (allows, e.g., symbolic coefficients).
            return Polynomial({v: c * other for v, c in self})

    def __imul__(self, other):
        return self * other

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):

        # poly ** 0 = 1, the case 0 ** 0 is left undefined.
        if power == 0:
            if len(self) == 0:
                raise ValueError('Undefined result for 0 ** 0.')
            vector_type = type(self.vectors()[0])
            return Polynomial({vector_type({}): 1})

        # Fall back to the multiplication method.
        return prod([self] * (power - 1), start=self)

    def derivative(self, variable):
        return sum(v.derivative(variable) * c for v, c in self)

    def jacobian(self, variables):
        return [self.derivative(v) for v in variables]

    def integral(self, variable):
        return sum(v.integral(variable) * c for v, c in self)

    def definite_integral(self, variables, lbs, ubs):
        if not len(variables) == len(lbs) == len(ubs):
            raise ValueError(f'integration range and variables have different lenghts.')
        integral = self
        for v, lb, ub in zip(variables, lbs, ubs):
            integral = integral.integral(v)
            integral = integral.substitute({v: ub}) - integral.substitute({v: lb})
        return integral

    def in_chebyshev_basis(self):
        return sum(v.in_chebyshev_basis() * c for v, c in self)

    def in_monomial_basis(self):
        return sum(v.in_monomial_basis() * c for v, c in self)

    def __repr__(self):

        # Represent polynomial as 0 if all the coefficients are 0.
        if len(self) == 0:
            return '0'

        # Add one addend per time to the representation.
        r = ''
        for vector, coef in self:

            # Do not represent plus sign if first addend or negative sign.
            if len(r) > 0 and optimistic(coef, gt, 0):
                r += '+'

            # Just represent the coefficient if vector is 1.
            if len(vector) == 0:
                r += str(coef)
            else:

                # Just represent - if coefficient is -1.
                if pessimistic(coef, eq, - 1):
                    r += '-'

                # Represent coefficient if different from +1.
                elif optimistic(coef, ne, 1):
                    r += str(coef)

                # Add representation of vector.
                r += vector.__repr__()

        return r
            
    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    def vectors(self):
        return list(self.coef_dict)

    def variables(self):
        return list(set(var for vec in self.vectors() for var in vec.variables()))

    def coefficients(self):
        return list(self.coef_dict.values())

    def degree(self):
        return max(v.degree() for v in self.vectors()) if len(self) > 0 else 0

    def is_odd(self):
        return all(v.is_odd() for v in self.vectors()) if len(self) > 0 else False

    def is_even(self):
        return all(v.is_even() for v in self.vectors())

    # ToDo: find a way to get rid of this method (currently needed bcs of definite_integral).
    def to_scalar(self):
        if self.degree() > 0:
            raise RuntimeError(f'polynomial cannot be converted to scalar, it has degree {self.degree()}.')
        return 0 if len(self) == 0 else self.coefficients()[0]

    @classmethod
    def quadratic_form(cls, basis, Q):
        p = cls({})
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis[i:]):
                j += i
                coef = 1 if i == j else 2
                p += (bi * bj) * (Q[i, j] * coef)
        return p

    @staticmethod
    def _verify_vectors(vectors):
        vector_types = set(type(v) for v in vectors)
        if len(vector_types) > 1:
            raise TypeError(f'basis vectors must have same type, got {t.__name__ for t in vector_types}.')
        elif len(vector_types) == 1:
            vector_type = list(vector_types)[0]
            if not issubclass(vector_type, poly.BasisVector):
                raise TypeError(f'basis vectors must be subclasses of BasisVector, got {vector_type.__name__}')

def pessimistic(a, op, b):
    return isinstance(a, Number) and isinstance(b, Number) and op(a, b)

def optimistic(a, op, b):
    return not isinstance(a, Number) or not isinstance(b, Number) or op(a, b)
