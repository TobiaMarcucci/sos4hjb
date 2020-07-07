import numpy as np

from sos4hjb.polynomials.variable import Variable

class BasisVector:
    '''
    Element of the basis of a polynomial.

    Attributes
    ----------
    power_dict : dict (key : Variable, value : int)
        Dictionary that maps each variable to its power.
    '''
    
    def __init__(self, power_dict):
        self.power_dict = {v: p for v, p in power_dict.items() if p != 0}
    
    def __getitem__(self, variable):
        return self.power_dict[variable] if variable in self.power_dict else 0
        
    def __setitem__(self, variable, power):
        if power == 0:
            self.power_dict.pop(variable, None)
        else:
            self.power_dict[variable] = power

    def __len__(self):
        return len(self.power_dict)

    def __iter__(self):
        return iter(self.power_dict.items())

    def __hash__(self):
        return hash(self._hash_tuple())

    def __eq__(self, vector):
        return self._hash_tuple() == vector._hash_tuple()
    
    def __ne__(self, vector):
        return not self == vector

    def _hash_tuple(self):
        assert not 0 in self.power_dict.values()
        hash_list = sorted((v.name, v.index, p) for v, p in self)
        hash_list.append(type(self).__name__)
        return tuple(hash_list)

    def __repr__(self):
        if len(self) == 0:
            return '1'
        return ''.join(self._repr(v, p) for v, p in self)

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @property
    def degree(self):
        return sum(self.power_dict.values())

    @staticmethod
    def _repr(variable, power):
        return f'({variable}, {power}) '
