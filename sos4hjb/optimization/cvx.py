import cvxpy as cp

from sos4hjb.polynomials import Polynomial
from sos4hjb.optimization import SosProgramParent

class SosProgram(SosProgramParent):
    
    def __init__(self):
        self.constraints = []
        self.cost = 0
        self.value = None
        
    def add_variables(self, size, name='c'):
        return cp.Variable(size, name)

    def add_psd_variable(self, size, name='Q'):
        gram = cp.Variable((size, size), name, symmetric=True)
        cons = gram >> 0
        self.constraints.append(cons)
        return gram, cons
    
    def add_linear_constraint(self, cons):
        self.constraints.append(cons)

    def add_linear_cost(self, expr):
        self.cost += expr
        
    def solve(self):
        prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        prob.solve(solver='MOSEK')
        self.value = prob.value

    def minimum(self):
        return self.value

    def substitute_minimizer(self, expr):
        if isinstance(expr, Polynomial):
            return Polynomial({v: c.value for v, c in expr})
        else:
            return expr.value
