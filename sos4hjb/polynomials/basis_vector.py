import numpy as np

from sos4hjb.polynomials.variable import Variable

class BasisVector:
    
    def __init__(self, power_dict):
        lgtm_power_dict(power_dict)
        self.power_dict = power_dict
    
    def __getitem__(self, v):
        is_variable(v)
        if v in self.power_dict:
            return self.power_dict[v]
        else:
            return 0
        
    def __setitem__(self, v, p):
        is_variable(v)
        is_power(p)
        self.power_dict[v] = p

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
    def _repr(v, p):
        return f'({v}, {p}) '

def is_variable(v):
    if not isinstance(v, Variable):
        raise ValueError(f'''Variables must be of type Variable, got
            {type(v)}.''')

def is_power(p):
    power_type = (int, np.int64)
    if not isinstance(p, power_type):
        raise ValueError(f'''Powers must be of any of these types:
            {power_type}. Got {type(p)}.''')
    if p < 0:
        raise ValueError(f'Powers must be nonnegative, got {p}.')

def lgtm_power_dict(power_dict):
    if not isinstance(power_dict, dict):
        raise ValueError(f'''power_dict must be a dictionary, got
            {type(power_dict)}.''')
    for v in power_dict.keys():
        is_variable(v)
    for p in power_dict.values():
        is_power(p)
