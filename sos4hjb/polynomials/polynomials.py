import numpy as np

class Vector:
    
    def __init__(self, power_dict):
        self._test_power_dict(power_dict)
        self.power_dict = power_dict
    
    def __getitem__(self, v):
        _test_variable(v)
        if v in self.power_dict:
            return self.power_dict[v]
        else:
            return 0
        
    def __setitem__(self, v, p):
        _test_variable(v)
        _test_power(p)
        self.power_dict[v] = p

    # ToDo: did I need this somewhere?
    def __iter__(self):
        return iter(self.power_dict.items())
    
    def __len__(self):
        return len(self.power_dict)

    def __lt__(self, vector):
        self_list, vector_list = self._power_dicts_to_lists(vector)
        return self_list < vector_list
    
    def __gt__(self, vector):
        self_list, vector_list = self._power_dicts_to_lists(vector)
        return self_list > vector_list
    
    def __le__(self, vector):
        return not self > vector
    
    def __ge__(self, vector):
        return not self < vector
    
    def __eq__(self, vector):
        self_list, vector_list = self._power_dicts_to_lists(vector)
        return self_list == vector_list
    
    def __ne__(self, vector):
        return not self == vector

    def _power_dicts_to_lists(self, vector):
        vs = self._merge_variables_with(vector)
        self_list = [self[v] for v in vs]
        vector_list = [vector[v] for v in vs]
        return self_list, vector_list

    def _merge_variables_with(self, vector):
        return sorted(set(list(self.power_dict) + list(vector.power_dict)))

    def _test_power_dict(self, power_dict):
        if not isinstance(power_dict, dict):
            raise ValueError(f'''power_dict must be a dictionary, got
                {power_dict.__class__}.''')
        for v in power_dict.keys():
            _test_variable(v)
        for p in power_dict.values():
            _test_power(p)
    
    def _test_evaluation_dict(self, evaluation_dict):
        for v in evaluation_dict.keys():
            _test_variable(v)
        if not all(v in evaluation_dict for v in self.power_dict):
            raise ValueError('''The evaluation dictionary must assign a value to
                each variable.''')

    def __repr__(self):
        if len(self) == 0:
            return '1'
        return ''.join(self._repr(v, p) for v, p in self)

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @staticmethod
    def _repr(v, p):
        return f'({v}, {p}) '

    @property
    def degree(self):
        return sum(self.power_dict.values())

def _test_variable(v):
    if not isinstance(v, str):
        raise ValueError(f'Variables must be strings, got {v.__class__}.')

def _test_power(p):
    power_type = (int, np.int64)
    if not isinstance(p, power_type):
        raise ValueError(f'''Powers must be of any of this type: {power_type}.
            Got {p.__class__}.''')
    if p < 0:
        raise ValueError(f'Powers must be nonnegative, got {p}.')

from copy import deepcopy

class Monomial(Vector):

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        self._test_evaluation_dict(evaluation_dict)
        return np.prod([value ** self[v] for v, value in evaluation_dict.items()])

    def __mul__(self, monomial):
        if not self.__class__ is type(monomial):
            raise ValueError(f'''Monomials can only be multiplied with
                monomials, got {type(monomial)}.''')
        vs = self._merge_variables_with(monomial)
        power_dict = {v: self[v] + monomial[v] for v in vs}
        return Polynomial([1], [Monomial(power_dict)])

    def derivative(self, v):
        c = self[v]
        if c == 0:
            monomial = Monomial({})
        else:
            monomial = deepcopy(self)
            monomial[v] -= 1
        return Polynomial([c], [monomial])

    def primitive(self, v):
        c = 1 / (self[v] + 1)
        monomial = deepcopy(self)
        monomial[v] += 1
        return Polynomial([c], [monomial])

    @staticmethod
    def _repr(v, p):
        return f'{v}^{{{p}}} '

from mpmath import chebyt
from itertools import product

class Chebyshev(Vector):

    def __init__(self, power_dict):
        super().__init__(power_dict)

    def __call__(self, evaluation_dict):
        self._test_evaluation_dict(evaluation_dict)
        return np.prod([float(chebyt(self[v], value)) for v, value in evaluation_dict.items()])

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
        chebyshevs = [Chebyshev({v: ps[i] for i, v in enumerate(vs)}) for ps in product(*product_pow)]

        return Polynomial(coefficients, chebyshevs)

    def derivative(self, v):
        _test_variable(v)

        # Derivative of 1 is 0.
        p = self[v]
        if p == 0:
            return Polynomial([0], Chebyshev({}))

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
        _test_variable(v)
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

class Polynomial:
    coefficient_type = (float, int, np.float64, np.int64) # Add Drake/Mosek optimization variables.
    vector_type = (Monomial, Chebyshev)

    def __init__(self, coefficients, basis):
        self._test_coefficients_and_basis(coefficients, basis)

        # Clean coefficients and basis from duplicate vectors.
        uniques = [basis.index(v) for v in basis]
        basis = [basis[i] for i in sorted(set(uniques))]
        duplicates = [np.where(np.array(uniques) == i)[0] for i in range(len(basis))]
        coefficients = [sum(coefficients[j] for j in i) for i in duplicates]

        # Remove zeros for the coefficents, and store.
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
        if isinstance(other, self.coefficient_type):
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
        _test_power(p)

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

    def _test_coefficients_and_basis(self, coefficients, basis):
        if len(coefficients) != len(basis):
            raise ValueError(f'''The coefficents and the basis must have the
                same length. Got coefficients of length {len(coefficients)}, and
                basis of lenght {len(basis)}.''')
        if len(set(type(vector) for vector in basis)) > 1:
            raise ValueError(f'''All the vectors in the basis must be of the
                same type, got {[vector.__class__ for vector in basis]}.''')
        for coefficient in coefficients:
            if not isinstance(coefficient, self.coefficient_type):
                raise ValueError(f'''Coefficients must be of any of this type:
                    {self.coefficient_type}, got {coefficient.__class__}.''')
        for vector in basis:
            if not isinstance(vector, self.vector_type):
                raise ValueError(f'''Vectors must be of any of this type:
                    {self.vector_type}, got {vector.__class__}.''')
