class Variable:
    '''
    Elementary variable for a polynomial.

    Attributes
    ----------
    name : str
        Name of the variable.
    index : int (>= 0)
        Index of the variable. If equal to 0, the variable will have no index.
    '''

    def __init__(self, name, index=0):
        self._verify_name(name)
        self._verify_index(index)
        self.name = name
        self.index = index

    def __hash__(self):
        return hash((self.name, self.index))

    def __eq__(self, variable):
        same_type = type(self) == type(variable)
        same_name = self.name == variable.name
        same_index = self.index == variable.index
        return same_type and same_name and same_index
    
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
