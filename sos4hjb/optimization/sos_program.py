from pydrake.all import MathematicalProgram, Expression

from sos4hjb.polynomials import Polynomial

class SosProgram(MathematicalProgram):
    
    def __init__(self):
        super().__init__()

    def NewFreePolynomial(self, basis, name='c'):
        c = self.NewContinuousVariables(len(basis), name)
        p = Polynomial(dict(zip(basis, c)))
        return  p, c

    def NewSosPolynomial(self, basis, name='Q'):
        Q = self.NewSymmetricContinuousVariables(len(basis), name)
        self.AddPositiveSemidefiniteConstraint(Q)
        p = Polynomial.quadratic_form(basis, Q)
        return p, Q
        
    def NewEvenDegreeSosPolynomial(self, basis, name='Q'):
        basis_e = [v for v in basis if v.is_even()]
        basis_o = [v for v in basis if v.is_odd()]
        p_e, Q_e = self.NewSosPolynomial(basis_e, name + '_{e}')
        p_o, Q_o = self.NewSosPolynomial(basis_o, name + '_{o}')
        return p_e + p_o, [Q_e, Q_o]
    
    def AddSosConstraint(self, p, name='Q'):
        if p.degree() % 2:
            raise ValueError(f'SOS polynomials must have even degree, got degree {p.degree()}.')
        if len(p) > 0:
            basis = p.vectors()[0].construct_basis(p.variables(), p.degree() // 2)
        else:
            basis = []
        if p.is_even():
            p_sos, Q = self.NewEvenDegreeSosPolynomial(basis, name)
        else:
            p_sos, Q = self.NewSosPolynomial(basis, name)
        for coef in (p - p_sos).coefficients():
            self.AddLinearEqualityConstraint(coef == 0)
        return Q

    @staticmethod
    def evaluate_at_optimum(p, result):
        p_opt = Polynomial({})
        for v, c in p:
            c_opt = result.GetSolution(c)
            if isinstance(c_opt, Expression):
                c_opt = c_opt.Evaluate()
            p_opt[v] = c_opt
        return p_opt
