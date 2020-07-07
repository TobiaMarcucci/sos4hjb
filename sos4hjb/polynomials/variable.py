class Variable:
    '''
    Elementary variable for a polynomial.

    Attributes
    ----------
    name : str
        Name of the variable.
    index : int
        Index of the variable. If equal to 0, the variable will haveno index.
    '''

    def __init__(self, name, index=0):
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
