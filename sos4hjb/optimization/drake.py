from pydrake.all import MathematicalProgram, Expression, Solve

from sos4hjb.polynomials import Polynomial
from sos4hjb.optimization import SosProgramParent

class SosProgram(SosProgramParent, MathematicalProgram):
    
    def __init__(self):
        MathematicalProgram.__init__(self)
        self.result = None

    def add_variables(self, size, name='c'):
        return self.NewContinuousVariables(size, name)

    def add_psd_variable(self, size, name='Q'):
        gram = self.NewSymmetricContinuousVariables(size, name)
        cons = self.AddPositiveSemidefiniteConstraint(gram)
        return gram, cons

    def add_linear_constraint(self, cons):
        self.AddLinearConstraint(cons)

    def add_linear_cost(self, expr):
        self.AddLinearCost(expr)

    def solve(self):
        self.result = Solve(self)

    def minimum(self):
        if self.result is not None:
            return self.result.get_optimal_cost()

    def substitute_minimizer(self, expr):
        if isinstance(expr, Polynomial):
            p_opt = Polynomial({})
            for v, c in expr:
                c_opt = self.result.GetSolution(c)
                if isinstance(c_opt, Expression):
                    c_opt = c_opt.Evaluate()
                p_opt[v] = c_opt
            return p_opt
        else:
            return self.result.GetSolution(expr)
