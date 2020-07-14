import unittest
import numpy as np
from pydrake.all import Solve


from sos4hjb.optimization import SosProgram
from sos4hjb.polynomials import (Variable, MonomialVector, ChebyshevVector,
                                 Polynomial)

vector_types = (MonomialVector, ChebyshevVector)

class TestSosProgram(unittest.TestCase):

    x = Variable.multivariate('x', 2)
    zero = {xi: 0 for xi in x}
    one = {xi: 1 for xi in x}
    two = {xi: 2 for xi in x}

    def test_new_free_polynomial(self):

        for vector_type in vector_types:

            # Fit free polynomial in 3 points.
            prog = SosProgram()
            basis = vector_type.construct_basis(self.x, 3)
            p, c = prog.NewFreePolynomial(basis)
            prog.AddLinearConstraint(p(self.zero) == 0)
            prog.AddLinearConstraint(p(self.one) == 1)
            prog.AddLinearConstraint(p(self.two) == 2)
            result = Solve(prog)
            c_opt = result.GetSolution(c)
            p_opt = prog.evaluate_at_optimum(p, result)
            self.assertAlmostEqual(p_opt(self.zero), 0)
            self.assertAlmostEqual(p_opt(self.one), 1)
            self.assertAlmostEqual(p_opt(self.two), 2)

    def test_new_sos_polynomial(self):

        # Fit free polynomial in 2 points, and minimize value at a third.
        for vector_type in vector_types:

            # Normal polynomial.
            prog = SosProgram()
            basis = vector_type.construct_basis(self.x, 3)
            p, Q = prog.NewSosPolynomial(basis)
            prog.AddLinearCost(p(self.zero))
            prog.AddLinearConstraint(p(self.one) == 1)
            prog.AddLinearConstraint(p(self.two) == 2)
            result = Solve(prog)
            p_opt = prog.evaluate_at_optimum(p, result)
            self.assertAlmostEqual(p_opt(self.zero), 0)
            self.assertAlmostEqual(p_opt(self.one), 1)
            self.assertAlmostEqual(p_opt(self.two), 2)

            # Reconstruct polynomial from Gram matrix.
            Q_opt = result.GetSolution(Q)
            self.assertTrue(self._is_psd(Q_opt))
            p_opt_Q = Polynomial.quadratic_form(basis, Q_opt)
            self.assertAlmostEqual(p_opt, p_opt_Q)

            # Even polynomial.
            prog = SosProgram()
            p, Q = prog.NewEvenDegreeSosPolynomial(basis)
            prog.AddLinearCost(p(self.zero))
            prog.AddLinearConstraint(p(self.one) == 1)
            prog.AddLinearConstraint(p(self.two) == 2)
            result = Solve(prog)
            p_opt = prog.evaluate_at_optimum(p, result)
            self.assertAlmostEqual(p_opt(self.zero), 0)
            self.assertAlmostEqual(p_opt(self.one), 1)
            self.assertAlmostEqual(p_opt(self.two), 2)
            
            # Reconstruct polynomial from Gram matrices.
            Q_opt_e, Q_opt_o = [result.GetSolution(Qi) for Qi in Q]
            self.assertTrue(self._is_psd(Q_opt_e))
            self.assertTrue(self._is_psd(Q_opt_o))
            basis_e = [v for v in basis if v.is_even]
            basis_o = [v for v in basis if v.is_odd]
            p_opt_Q = Polynomial.quadratic_form(basis_e, Q_opt_e)
            p_opt_Q += Polynomial.quadratic_form(basis_o, Q_opt_o)
            self.assertAlmostEqual(p_opt, p_opt_Q)

    def test_add_sos_constraint(self):

        # Fit free polynomial in 2 points, and minimize value at a third.
        for vector_type in vector_types:

            # Normal polynomial.
            prog = SosProgram()
            basis = vector_type.construct_basis(self.x, 6)
            p, c = prog.NewFreePolynomial(basis)
            Q = prog.AddSosConstraint(p)
            prog.AddLinearCost(p(self.zero))
            prog.AddLinearConstraint(p(self.one) == 1)
            prog.AddLinearConstraint(p(self.two) == 2)
            result = Solve(prog)
            p_opt = prog.evaluate_at_optimum(p, result)
            self.assertAlmostEqual(p_opt(self.zero), 0)
            self.assertAlmostEqual(p_opt(self.one), 1)
            self.assertAlmostEqual(p_opt(self.two), 2)

            # Reconstruct polynomial from Gram matrix.
            Q_opt = result.GetSolution(Q)
            self.assertTrue(self._is_psd(Q_opt))
            basis_half = vector_type.construct_basis(self.x, 3)
            p_opt_Q = Polynomial.quadratic_form(basis_half, Q_opt)
            self.assertAlmostEqual(p_opt, p_opt_Q)

            # Even polynomial.
            prog = SosProgram()
            basis = vector_type.construct_basis(self.x, 6, odd=False)
            p, c = prog.NewFreePolynomial(basis)
            Q = prog.AddSosConstraint(p)
            prog.AddLinearCost(p(self.zero))
            prog.AddLinearConstraint(p(self.one) == 1)
            prog.AddLinearConstraint(p(self.two) == 2)
            result = Solve(prog)
            p_opt = prog.evaluate_at_optimum(p, result)
            self.assertAlmostEqual(p_opt(self.zero), 0)
            self.assertAlmostEqual(p_opt(self.one), 1)
            self.assertAlmostEqual(p_opt(self.two), 2)

            # Reconstruct polynomial from Gram matrices.
            Q_opt_e, Q_opt_o = [result.GetSolution(Qi) for Qi in Q]
            self.assertTrue(self._is_psd(Q_opt_e))
            self.assertTrue(self._is_psd(Q_opt_o))
            basis = vector_type.construct_basis(self.x, 3)
            basis_e = [v for v in basis if v.is_even]
            basis_o = [v for v in basis if v.is_odd]
            p_opt_Q = Polynomial.quadratic_form(basis_e, Q_opt_e)
            p_opt_Q += Polynomial.quadratic_form(basis_o, Q_opt_o)
            self.assertAlmostEqual(p_opt, p_opt_Q)

            # Polynomial of odd degree.
            prog = SosProgram()
            basis = vector_type.construct_basis(self.x, 3)
            p, c = prog.NewFreePolynomial(basis)
            with self.assertRaises(ValueError):
                prog.AddSosConstraint(p)

            # Polynomial of length 0.
            prog = SosProgram()
            p = Polynomial({})
            Q = prog.AddSosConstraint(p)
            self.assertEqual(Q[0].shape, (0, 0))
            self.assertEqual(Q[1].shape, (0, 0))

    @staticmethod
    def _is_psd(Q):
        return all(np.linalg.eig(Q)[0] > 0)
