import numpy as np
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
        self.coef_dict = {vector: coef for vector, coef in coef_dict.items() if coef != 0}

    def __getitem__(self, vector):
        return self.coef_dict[vector] if vector in self.coef_dict else 0

    def __setitem__(self, vector, coef):
        if coef == 0:
            self.coef_dict.pop(vector, None)
        else:
            self.coef_dict[vector] = coef

    def __len__(self):
        return len(self.coef_dict)

    def __iter__(self):
        return iter(self.coef_dict.items())

    # Currently allows addition of monomials of different type.
    def __add__(self, poly):
        vectors = set(list(self.coef_dict) + list(poly.coef_dict))
        coef_dict = {vector: self[vector] + poly[vector] for vector in vectors}
        return Polynomial(coef_dict)

    def __iadd__(self, poly):
        return self + poly

    def __sub__(self, poly):
        return self + poly * (-1)

    def __isub__(self, poly):
        return self - poly

    # Broken multiplication of monomials of different type.
    def __mul__(self, other):

        # If multiplication by a scalar.
        coef_type = (float, int, np.float64, np.int64)
        if isinstance(other, coef_type):
            return Polynomial({vector: coef * other for vector, coef in self})

        # If multiplication by another polynomial.
        return Polynomial({vs * vo: cs * co for vs, cs in self for vo, co in other})

    def __imul__(self, poly):
        return self * poly

    def __pow__(self, power):

        # 0 ** power = 0.
        if len(self) == 0:
            return Polynomial({})

        # poly ** 0 = 1 (note that 0 ** 0 is already ruled out).
        if power == 0:
            vector_type = type(list(self.coef_dict)[0])
            return Polynomial({vector_type({}): 1})

        # Fall back to the multiplication method.
        poly = deepcopy(self)
        for i in range(power - 1):
            poly *= self
        return poly

    def derivative(v):
        pass

    def primitive(v):
        pass

    def __repr__(self):

        # Represent polynomial as 0 if all the coefficients are 0.
        assert not 0 in self.coef_dict.values()
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
