class Variable:

    def __init__(self, name, index=None):
        lgtm_name(name)
        lgtm_index(index)
        self.name = name
        self.index = index

    def __hash__(self):
        return hash((self.name, self.index))

    def __lt__(self, v):
        self_index = -1 if self.index is None else self.index
        v_index = -1 if v.index is None else v.index
        return self.name < v.name or (self.name == v.name and self_index < v_index)

    def __gt__(self, v):
        self_index = -1 if self.index is None else self.index
        v_index = -1 if v.index is None else v.index
        return self.name > v.name or (self.name == v.name and self_index > v_index)
    
    def __le__(self, v):
        return not self > v
    
    def __ge__(self, v):
        return not self < v
    
    def __eq__(self, other):
        return self.name == other.name and self.index == other.index
    
    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        representation = self.name
        if self.index is not None:
            representation += f'_{{{self.index}}}'
        return representation

    def _repr_latex_(self):
        return '$' + self.__repr__() + '$'

    @staticmethod
    def multivariate(name, size):
        
        return [Variable(name, index=i) for i in range(size)]

def lgtm_name(name):
    if not isinstance(name, str):
        raise ValueError(f'''The name of an variable must be a string,
            got {type(name)}.''')

def lgtm_index(index):
    if index is not None:
        if not isinstance(index, int):
            raise ValueError(f'''The index of a variable must be an integer
                or None, got {type(index)}.''')
        if index < 0:
            raise ValueError(f'''If integer, the index of a variable must be
                nonnegative, got {index}.''')
