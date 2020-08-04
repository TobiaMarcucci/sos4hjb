import unittest
import numpy as np

from sos4hjb.optimization.cvx import SosProgram as CvxSosProgram
from sos4hjb.optimization.drake import SosProgram as DrakeSosProgram
from sos4hjb.polynomials import (Variable, MonomialVector, ChebyshevVector,
                                 Polynomial)

SosPrograms = (CvxSosProgram, DrakeSosProgram)
Vectors = (MonomialVector, ChebyshevVector)

class TestSosProgram(unittest.TestCase):

    x = Variable.multivariate('x', 2)
    zero = {xi: 0 for xi in x}
    one = {xi: 1 for xi in x}
    two = {xi: 2 for xi in x}

    def test_new_free_polynomial(self):

        for Vector, SosProgram in zip(Vectors, SosPrograms):

            # Fit free polynomial in 3 points.
            prog = SosProgram()
            basis = Vector.construct_basis(self.x, 3)
            poly, coef = prog.add_polynomial(basis)
            prog.add_linear_constraint(poly(self.zero) == 0)
            prog.add_linear_constraint(poly(self.one) == 1)
            prog.add_linear_constraint(poly(self.two) == 2)
            prog.solve()
            coef_opt = prog.substitute_minimizer(coef)
            poly_opt = prog.substitute_minimizer(poly)
            self.assertAlmostEqual(poly_opt(self.zero), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.one), 1, places=4)
            self.assertAlmostEqual(poly_opt(self.two), 2, places=4)

    def test_new_sos_polynomial(self):

        # Fit free polynomial in 2 points, and minimize value at a third.
        for Vector, SosProgram in zip(Vectors, SosPrograms):

            # Normal polynomial.
            prog = SosProgram()
            basis = Vector.construct_basis(self.x, 3)
            poly, gram, cons = prog.add_sos_polynomial(basis)
            prog.add_linear_cost(poly(self.zero))
            prog.add_linear_constraint(poly(self.one) == 1)
            prog.add_linear_constraint(poly(self.two) == 2)
            prog.solve()
            poly_opt = prog.substitute_minimizer(poly)
            self.assertAlmostEqual(prog.minimum(), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.zero), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.one), 1, places=4)
            self.assertAlmostEqual(poly_opt(self.two), 2, places=4)

            # Reconstruct polynomial from Gram matrix.
            gram_opt = prog.substitute_minimizer(gram)
            self.assertTrue(self._is_psd(gram_opt))
            poly_opt_gram = Polynomial.quadratic_form(basis, gram_opt)
            self.assertAlmostEqual(poly_opt, poly_opt_gram)

            # Even polynomial.
            prog = SosProgram()
            poly, gram, cons = prog.add_even_sos_polynomial(basis)
            prog.add_linear_cost(poly(self.zero))
            prog.add_linear_constraint(poly(self.one) == 1)
            prog.add_linear_constraint(poly(self.two) == 2)
            prog.solve()
            poly_opt = prog.substitute_minimizer(poly)
            self.assertAlmostEqual(prog.minimum(), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.zero), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.one), 1, places=4)
            self.assertAlmostEqual(poly_opt(self.two), 2, places=4)
            
            # Reconstruct polynomial from Gram matrices.
            gram_opt_e, gram_opt_o = [prog.substitute_minimizer(gi) for gi in gram]
            self.assertTrue(self._is_psd(gram_opt_e))
            self.assertTrue(self._is_psd(gram_opt_o))
            basis_e = [v for v in basis if v.is_even()]
            basis_o = [v for v in basis if v.is_odd()]
            poly_opt_gram = Polynomial.quadratic_form(basis_e, gram_opt_e)
            poly_opt_gram += Polynomial.quadratic_form(basis_o, gram_opt_o)
            self.assertAlmostEqual(poly_opt, poly_opt_gram)

    def test_add_sos_constraint(self):

        # Fit free polynomial in 2 points, and minimize value at a third.
        for Vector, SosProgram in zip(Vectors, SosPrograms):

            # Normal polynomial.
            prog = SosProgram()
            basis = Vector.construct_basis(self.x, 6)
            poly, coef = prog.add_polynomial(basis)
            gram = prog.add_sos_constraint(poly)[1]
            prog.add_linear_cost(poly(self.zero))
            prog.add_linear_constraint(poly(self.one) == 1)
            prog.add_linear_constraint(poly(self.two) == 2)
            prog.solve()
            poly_opt = prog.substitute_minimizer(poly)
            self.assertAlmostEqual(prog.minimum(), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.zero), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.one), 1, places=4)
            self.assertAlmostEqual(poly_opt(self.two), 2, places=4)

            # Reconstruct polynomial from Gram matrix.
            gram_opt = prog.substitute_minimizer(gram)
            self.assertTrue(self._is_psd(gram_opt))
            basis_half = Vector.construct_basis(self.x, 3)
            poly_opt_gram = Polynomial.quadratic_form(basis_half, gram_opt)
            self.assertAlmostEqual(poly_opt, poly_opt_gram)

            # Even polynomial.
            prog = SosProgram()
            basis = Vector.construct_basis(self.x, 6, odd=False)
            poly, coef = prog.add_polynomial(basis)
            gram = prog.add_sos_constraint(poly)[1]
            prog.add_linear_cost(poly(self.zero))
            prog.add_linear_constraint(poly(self.one) == 1)
            prog.add_linear_constraint(poly(self.two) == 2)
            prog.solve()
            poly_opt = prog.substitute_minimizer(poly)
            self.assertAlmostEqual(prog.minimum(), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.zero), 0, places=4)
            self.assertAlmostEqual(poly_opt(self.one), 1, places=4)
            self.assertAlmostEqual(poly_opt(self.two), 2, places=4)

            # Reconstruct polynomial from Gram matrices.
            gram_opt_e, gram_opt_o = [prog.substitute_minimizer(gi) for gi in gram]
            self.assertTrue(self._is_psd(gram_opt_e))
            self.assertTrue(self._is_psd(gram_opt_o))
            basis = Vector.construct_basis(self.x, 3)
            basis_e = [v for v in basis if v.is_even()]
            basis_o = [v for v in basis if v.is_odd()]
            poly_opt_gram = Polynomial.quadratic_form(basis_e, gram_opt_e)
            poly_opt_gram += Polynomial.quadratic_form(basis_o, gram_opt_o)
            self.assertAlmostEqual(poly_opt, poly_opt_gram)

            # Polynomial of odd degree.
            prog = SosProgram()
            basis = Vector.construct_basis(self.x, 3)
            poly, c = prog.add_polynomial(basis)
            with self.assertRaises(ValueError):
                prog.add_sos_constraint(poly)

            # Polynomial of length 0.
            prog = SosProgram()
            poly = Polynomial({})
            with self.assertRaises(ValueError):
                prog.add_sos_constraint(poly)

    @staticmethod
    def _is_psd(A):
        return all(np.linalg.eig(A)[0] > 0)
