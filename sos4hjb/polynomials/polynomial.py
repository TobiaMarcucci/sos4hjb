import numpy as np
from copy import deepcopy

from sos4hjb.polynomials.basis_vector import BasisVector, is_power

# ToDo: support Drake/Mosek optimization variables.
coefficient_type = (float, int, np.float64, np.int64)

class Polynomial:

    def __init__(self, coefficients, basis):
        lgtm_coefficients_and_basis(coefficients, basis)

        # Remove duplicate basis vectors.
        uniques = [basis.index(v) for v in basis]
        basis = [basis[i] for i in sorted(set(uniques))]
        duplicates = [np.where(np.array(uniques) == i)[0] for i in range(len(basis))]
        coefficients = [sum(coefficients[j] for j in i) for i in duplicates]

        # Remove zeros from the coefficents, sort, and store.
        nonzeros = np.where(np.array(coefficients) != 0)[0]
        self.basis = [basis[i] for i in nonzeros]
        self.coefficients = [coefficients[i] for i in nonzeros]

    def __len__(self):
        return len(self.coefficients)

    def __iter__(self):
        return zip(self.coefficients, self.basis)

    def __add__(self, poly):
        coefficients = self.coefficients + poly.coefficients
        basis = self.basis + poly.basis
        return Polynomial(coefficients, basis)

    def __iadd__(self, poly):
        return self + poly

    def __sub__(self, poly):
        return self + poly * (-1)

    def __isub__(self, poly):
        return self - poly

    def __mul__(self, other):

        # If multiplication by a scalar.
        if isinstance(other, coefficient_type):
            prod = deepcopy(self)
            prod.coefficients = [c * other for c in prod.coefficients]
            return prod

        # If multiplication by another polynomial.
        elif isinstance(other, self.__class__):
            poly = Polynomial([], [])
            for cs, vs in self:
                for co, vo, in other:
                    poly += (vs * vo) * (cs * co)
            return poly

        # Raise error if type is unknown.
        else:
            raise ValueError(f'''Cannot multiply a polynomial by a
            {other.__class__}.''')

    def __imul__(self, poly):
        return self * poly

    def __pow__(self, p):
        is_power(p)

        # 0 ** p = 0.
        if len(self) == 0:
            return Polynomial([], [])

        # poly ** 0 = 1.
        if p == 0:
            return Polynomial([1], [type(self.basis[0])({})])

        # Fall back to the multiplication method.
        poly = deepcopy(self)
        for i in range(p - 1):
            poly *= self
        return poly

    # ToDo: avoid +- when coefficient is negative.
    def __repr__(self):

        # Add one addend per time to the representation.
        addends = []
        for coefficient, vector in zip(self.coefficients, self.basis):

            # If coefficient is zero, do not represent addend.
            if coefficient == 0:
                continue

            # Represent coefficient if different from one, or if vector
            # multiindex is empty.
            addend = ''
            if coefficient != 1 or (coefficient == 1 and len(vector) == 0):
                addend += str(coefficient) + ' '

            # Do not represent vector if multiindex is empty.
            if len(vector) > 0:
                addend += vector.__repr__()
            addends.append(addend)

        # Represent polynomial as 0 if all the coefficients are zero.
        if len(addends) == 0:
            return '0'

        # Sum all the addends.
        else:
            return ' + '.join(addends)
            
    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    def derivative(v):
    	pass

    def primitive(v):
    	pass

def lgtm_coefficients_and_basis(coefficients, basis):

    if len(coefficients) != len(basis):
        raise ValueError(f'''The coefficents and the basis must have the
            same length. Got coefficients of length {len(coefficients)}, and
            basis of lenght {len(basis)}.''')
    if len(set(type(v) for v in basis)) > 1:
        raise ValueError(f'''All the vectors in the basis must be of the
            same type, got {[type(v) for v in basis]}.''')
    for c in coefficients:
        if not isinstance(c, coefficient_type):
            raise ValueError(f'''Coefficients must be of any of this type:
                {coefficient_type}, got {type(c)}.''')
    for v in basis:
        if not issubclass(type(v), BasisVector):
            raise ValueError(f'''Vectors must be subclasses of BasisVector,
                got {type(v)}.''')
