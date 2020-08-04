from sos4hjb.polynomials import Polynomial

class SosProgramParent:
    '''
    Derived classes must implement the following methods:
        - add_variables(size, name)
        - add_psd_variable(size, name)
        - add_linear_constraint(constraint)
        - add_linear_cost(expression)
        - solve()
        - minimum()
        - substitute_minimizer(expression)
    '''
    
    def add_polynomial(self, basis, name='c'):
        coef = self.add_variables(len(basis), name)
        poly = Polynomial(dict(zip(basis, coef)))
        return  poly, coef

    def add_sos_polynomial(self, basis, name='Q'):
        gram, cons = self.add_psd_variable(len(basis), name)
        poly = Polynomial.quadratic_form(basis, gram)
        return poly, gram, cons
    
    def add_even_sos_polynomial(self, basis, name='Q'):

        # Split basis in even and odd vectors.
        basis_e = [v for v in basis if v.is_even()]
        basis_o = [v for v in basis if v.is_odd()]

        # Add two separate SOS constraints.
        poly_e, gram_e, cons_e = self.add_sos_polynomial(basis_e, name + '_{e}')
        poly_o, gram_o, cons_o = self.add_sos_polynomial(basis_o, name + '_{o}')

        # Gather the outputs.
        poly = poly_e + poly_o
        gram = [gram_e, gram_o]
        cons = [cons_e, cons_o]

        return poly, gram, cons
        
    def add_sos_constraint(self, p, name='Q'):

        # Raise error if polynomial has odd degree.
        if p.degree() % 2:
            raise ValueError(f'SOS polynomials must have even degree, got degree {p.degree()}.')

        # Construct basis for auxiliary SOS polynomial. If the given polynomial
        # has zero length then it is zero, and the basis is empty list.
        if len(p) == 0:
            raise ValueError(f'The given polynomial is zero, cannot add SOS constraint.')
        else:
            vector = p.vectors()[0]
            basis_degree = p.degree() // 2
            basis = vector.construct_basis(p.variables(), basis_degree)

        # Exploit even symmetry if present.
        if p.is_even():
            p_sos, gram_sos, cons_sos = self.add_even_sos_polynomial(basis, name)
        else:
            p_sos, gram_sos, cons_sos = self.add_sos_polynomial(basis, name)

        # Constrain the coefficients of the given and auxiliary polynomials.
        p_diff = p - p_sos
        cons_eq = [coef == 0 for coef in p_diff.coefficients()]
        for cons in cons_eq:
            self.add_linear_constraint(cons)

        return p_sos, gram_sos, cons_sos, cons_eq
