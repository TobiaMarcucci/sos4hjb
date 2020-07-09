import numpy as np

class Variable:
    '''
    Elementary variable for a polynomial.

    Attributes
    ----------
    name : str
        Name of the variable.
    index : int
        Index of the variable. If equal to 0, the variable will have no index.
    '''

    def __init__(self, name, index=0):
        _raise_if_not_nonnegative_int(index, 'index')
        self.name = name
        self.index = index

    def __hash__(self):
        return hash((self.name, self.index))

    def __eq__(self, variable):
        return self.name == variable.name and self.index == variable.index
    
    def __ne__(self, variable):
        return not self == variable

    def __repr__(self):
        representation = self.name
        if self.index != 0:
            representation += f'_{{{self.index}}}'
        return representation

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @staticmethod
    def multivariate(name, size): 
        return [Variable(name, index=i) for i in range(1, size + 1)]

def _raise_if_not_nonnegative_int(x, name):
    if not isinstance(x, (int, np.int64)):
        raise TypeError(f'{name} must be integer, got {type(x).__name__}.')
    if x < 0:
        raise ValueError(f'{name} must be nonnegative, got {x}.')
