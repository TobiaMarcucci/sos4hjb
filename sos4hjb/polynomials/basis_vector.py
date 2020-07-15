from math import prod
from itertools import combinations

import sos4hjb.polynomials as poly

class BasisVector:
    '''
    Element of the basis of a polynomial.

    Attributes
    ----------
    power_dict : dict (key : Variable, value : int)
        Dictionary that maps each variable to its power.
    '''
    
    def __init__(self, power_dict):
        for variable, power in power_dict.items():
            self._verify_variable(variable)
            self._verify_power(power)
        self.power_dict = {v: p for v, p in power_dict.items() if p != 0}

    def __call__(self, evaluation_dict):
        coefficient = prod([self._call_univariate(self[var], val) for var, val in evaluation_dict.items()])
        vector = type(self)({v: p for v, p in self if not v in evaluation_dict})
        return poly.Polynomial({vector: coefficient})

    def __getitem__(self, variable):
        self._verify_variable(variable)
        return self.power_dict[variable] if variable in self.power_dict else 0
        
    def __setitem__(self, variable, power):
        self._verify_variable(variable)
        self._verify_power(power)
        if power == 0:
            self.power_dict.pop(variable, None)
        else:
            self.power_dict[variable] = power

    def __hash__(self):
        return hash(self._hash_tuple())

    def __eq__(self, vector):
        return self._hash_tuple() == vector._hash_tuple()
    
    def __ne__(self, vector):
        return not self == vector

    def _hash_tuple(self):
        assert not 0 in self.powers
        hash_list = sorted((v.name, v.index, p) for v, p in self)
        hash_list.append(type(self).__name__)
        return tuple(hash_list)

    def __len__(self):
        return len(self.power_dict)

    def __iter__(self):
        return iter(self.power_dict.items())

    def __repr__(self):
        if len(self) == 0:
            return '1'
        return ''.join(self._repr(v, p) for v, p in self)

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @property
    def variables(self):
        return list(self.power_dict)

    @property
    def powers(self):
        return list(self.power_dict.values())

    @property
    def degree(self):
        return sum(self.powers)

    @property
    def is_even(self):
        return self.degree % 2 == 0

    @property
    def is_odd(self):
        return not self.is_even

    @classmethod
    def _vectors_of_degree(cls, variables, degree):
        positions = len(variables) + degree - 1
        breaks = len(variables) - 1
        vectors = []
        for c in combinations(range(positions), breaks):
            c = (- 1, *c, positions)
            powers = [p - q - 1 for q, p in zip(c, c[1:])]
            vectors.append(cls(dict(zip(variables, powers))))
        return vectors

    @classmethod
    def construct_basis(cls, variables, degree, even=True, odd=True):
        cls._verify_power(degree)
        vectors = []
        for d in range(degree + 1):
            if (even and d % 2 == 0) or (odd and d % 2):
                vectors += cls._vectors_of_degree(variables, d)
        return vectors

    @staticmethod
    def _repr(variable, power):
        return f'({variable},{power})'

    @ staticmethod
    def _verify_variable(variable):
        if not isinstance(variable, poly.Variable):
            raise TypeError(f'variable must be of Variable type, got {type(variable).__name__}.')

    @ staticmethod
    def _verify_power(power):
        if power % 1 or power < 0:
            raise ValueError(f'power must be a nonnegative integer, got {power}.')

    def _verify_multiplicand(self, vector):
        if not isinstance(vector, type(self)):
            raise TypeError(f'cannot multiply {type(self).__name__} with {type(vector).__name__}.')
