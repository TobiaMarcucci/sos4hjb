class Variable:
    '''
    Elementary variable for a polynomial.

    Attributes
    ----------
    name : str
        Name of the variable.
    index : int (>= 0)
        Index of the variable. If equal to 0, the variable will have no index.
    _hash : int
        Stored (and not computed on the fly) to accelerate comparisons.
    '''

    def __init__(self, name, index=0):
        self._verify_name(name)
        self._verify_index(index)
        self.name = name
        self.index = index
        self._hash = hash((self.name, self.index))

    def __hash__(self):
        return self._hash

    def __eq__(self, variable):
        return self._hash == variable._hash
    
    def __ne__(self, variable):
        return not self == variable

    def __repr__(self):
        representation = self.name
        if self.index != 0:
            representation += f'_{{{self.index}}}'
        return representation

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @classmethod
    def multivariate(cls, name, size):
        return [cls(name, index=i) for i in range(1, size + 1)]

    @ staticmethod
    def _verify_name(name):
        if not isinstance(name, str):
            raise TypeError(f'name must be a string, got {type(name).__name__}.')

    @ staticmethod
    def _verify_index(index):
        if index % 1 or index < 0:
            raise ValueError(f'index must be a nonnegative integer, got {index}.')
