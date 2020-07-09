import numpy as np
from math import prod
from copy import deepcopy

from sos4hjb.polynomials.basis_vector import BasisVector

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
        self.coef_dict = {v: c for v, c in coef_dict.items() if c != 0}

    def __getitem__(self, vector):
        return self.coef_dict[vector] if vector in self.coef_dict else 0

    def __setitem__(self, vector, coef):
        if coef == 0:
            self.coef_dict.pop(vector, None)
        else:
            self.coef_dict[vector] = coef
            self._verify_vectors(self.vectors)

    def __len__(self):
        return len(self.coef_dict)

    def __contains__(self, vector):
        return vector in self.coef_dict

    def __iter__(self):
        return iter(self.coef_dict.items())

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

        # Multiplication by a scalar.
        scalar = (float, int, np.float64, np.int64)
        if isinstance(other, scalar):
            return Polynomial({v: c * other for v, c in self})

        # Multiplication by a polynomial.
        return sum([(vs * vo) * (cs * co) for vs, cs in self for vo, co in other], Polynomial({}))

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

    def derivative(self, variable):
        return sum([v.derivative(variable) * c for v, c in self], Polynomial({}))

    def primitive(self, variable):
        return sum([v.primitive(variable) * c for v, c in self], Polynomial({}))

    def __repr__(self):

        # Represent polynomial as 0 if all the coefficients are 0.
        assert not 0 in self.coefficients
        if len(self) == 0:
            return '0'

        # Add one addend per time to the representation.
        addends = []
        for vector, coef in self:

            # Represent coefficient if different from 1, or if vector is 1.
            addend = '+' if coef > 0 and len(addends) > 0 else ''
            if coef != 1 or len(vector) == 0:
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
    def coefficients(self):
        return list(self.coef_dict.values())

    @property
    def degree(self):
        return max(v.degree for v in self.vectors)

    @staticmethod
    def _verify_vectors(vectors):
        for vector in vectors:
            if not issubclass(type(vector), BasisVector):
                raise TypeError(f'basis vectors must be subclasses of BasisVector.')
        vector_types = set(type(v).__name__ for v in vectors)
        if len(vector_types) > 1:
            raise TypeError(f'basis vectors must have same type, got {vector_types}.')
