from math import prod
from operator import eq, ne, gt
from numbers import Number

import sos4hjb.polynomials as poly

class Polynomial:
    '''
    Polynomial expressed as the linear combination of basis vectors.

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
            self._verify_vectors(self.vectors)

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
        return sum([v(evaluation_dict) * c for v, c in self], Polynomial({}))

    def __add__(self, poly):
        vectors = set(self.vectors + poly.vectors)
        return Polynomial({v: self[v] + poly[v] for v in vectors})

    def __iadd__(self, poly):
        return self + poly

    def __sub__(self, poly):
        return self + poly * (-1)

    def __isub__(self, poly):
        return self - poly

    def __mul__(self, other):

        # Multiplication by another polynomial.
        if isinstance(other, Polynomial):
            return sum([(vs * vo) * (cs * co) for vs, cs in self for vo, co in other], Polynomial({}))

        # Anything different from a Polynomial is treated as a scalar.
        else:
            return Polynomial({v: c * other for v, c in self})

    def __imul__(self, other):
        return self * other

    def __pow__(self, power):

        # poly ** 0 = 1, the case 0 ** 0 is left undefined.
        if power == 0:
            if len(self) == 0:
                raise ValueError('Undefined result for 0 ** 0.')
            vector_type = type(self.vectors[0])
            return Polynomial({vector_type({}): 1})

        # Fall back to the multiplication method.
        return prod([self] * (power - 1), start=self)

    def __round__(self, n=0):
        return Polynomial({v: round(c, n) for v, c in self})

    def __abs__(self):
        return Polynomial({v: abs(c) for v, c in self})

    def derivative(self, variable):
        return sum([v.derivative(variable) * c for v, c in self], Polynomial({}))

    def jacobian(self, variables):
        return [self.derivative(v) for v in variables]

    def integral(self, variable):
        return sum([v.integral(variable) * c for v, c in self], Polynomial({}))

    def definite_integral(self, variables, lbs, ubs):
        if not len(variables) == len(lbs) == len(ubs):
            raise ValueError(f'integration range and variables have different lenghts.')
        integral = self
        for v, lb, ub in zip(variables, lbs, ubs):
            integral = integral.integral(v)
            integral = integral({v: ub}) - integral({v: lb})
        return integral

    def __repr__(self):

        # Represent polynomial as 0 if all the coefficients are 0.
        if len(self) == 0:
            return '0'

        # Add one addend per time to the representation.
        addends = []
        for vector, coef in self:

            # Represent coefficient if different from 1, or if vector is 1.
            addend = '+' if len(addends) > 0 and optimistic(coef, gt, 0) else ''
            if optimistic(coef, ne, 1) or len(vector) == 0:
                addend += str(coef)

            # Do not represent vector if 1.
            if len(vector) > 0:
                addend += vector.__repr__()
            addends.append(addend)

        # Sum all the addends.
        return ''.join(addends)
            
    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @property
    def vectors(self):
        return list(self.coef_dict)

    @property
    def variables(self):
        return list(set(var for vec in self.vectors for var in vec.variables))

    @property
    def coefficients(self):
        return list(self.coef_dict.values())

    @property
    def degree(self):
        return max(v.degree for v in self.vectors) if len(self) > 0 else 0

    @property
    def is_odd(self):
        return all(v.is_odd for v in self.vectors) if len(self) > 0 else False

    @property
    def is_even(self):
        return all(v.is_even for v in self.vectors)

    @property
    def to_scalar(self):
        if self.degree > 0:
            raise RuntimeError(f'polynomial cannot be converted to scalar, it has degree {self.degree}.')
        return 0 if len(self) == 0 else self.coefficients[0]

    @classmethod
    def quadratic_form(cls, basis, Q):
        p = cls({})
        for i, bi in enumerate(basis):
            for j, bj in enumerate(basis):
                p += (bi * bj) * Q[i, j]
        return p

    @staticmethod
    def _verify_vectors(vectors):
        for vector in vectors:
            if not isinstance(vector, poly.BasisVector): # True also if subclass.
                raise TypeError(f'basis vectors must be subclasses of BasisVector, got {type(vector).__name__}')
        vector_types = set(type(v).__name__ for v in vectors)
        if len(vector_types) > 1:
            raise TypeError(f'basis vectors must have same type, got {vector_types}.')

def pessimistic(a, op, b):
    return isinstance(a, Number) and isinstance(b, Number) and op(a, b)

def optimistic(a, op, b):
    return not isinstance(a, Number) or not isinstance(b, Number) or op(a, b)
