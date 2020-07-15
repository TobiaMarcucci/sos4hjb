import unittest

from sos4hjb.polynomials import (Variable, MonomialVector, ChebyshevVector,
                                 Polynomial)

class TestChebyshevVector(unittest.TestCase):

    def test_call(self):

        # Partial evaluation.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        power_dict = {x: 5, y: 2, z: 3}
        m = ChebyshevVector(power_dict)
        x_eval = {x: - 2.1, y: 1.5}
        m_eval = ChebyshevVector({z: 3})
        c_eval = 16 * (- 2.1) ** 5 - 20 * (- 2.1) ** 3 + 5 * (- 2.1)
        c_eval *= 2 * 1.5 ** 2 - 1
        p = Polynomial({m_eval: c_eval})
        self.assertAlmostEqual(m(x_eval), p)

        # Complete evaluation.
        x_eval[z] = 5
        m_eval = ChebyshevVector({})
        c_eval *= 4 * 5 ** 3 - 3 * 5
        p = Polynomial({m_eval: c_eval})
        self.assertAlmostEqual(m(x_eval), p)

        # Cancellation.
        x_eval = {z: 0}
        p = Polynomial({})
        self.assertAlmostEqual(m(x_eval), p)

    def test_mul(self):

        # Bivariate times univariate.
        x = Variable('x')
        y = Variable('y')
        c0 = ChebyshevVector({x: 5, y: 3})
        c1 = ChebyshevVector({x: 2})
        p = c0 * c1
        self.assertEqual(len(p), 2)
        self.assertEqual(p[ChebyshevVector({x: 7, y: 3})], 1 / 2)
        self.assertEqual(p[ChebyshevVector({x: 3, y: 3})], 1 / 2)

        # Bivariate times bivariate.
        c1 = ChebyshevVector({x: 8, y: 1})
        p = c0 * c1
        self.assertEqual(len(p), 4)
        self.assertEqual(p[ChebyshevVector({x: 13, y: 4})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({x: 13, y: 2})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({x: 3, y: 4})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({x: 3, y: 2})], 1 / 4)

        # With zero power.
        c1 = ChebyshevVector({x: 5, y: 1})
        p = c0 * c1
        self.assertEqual(len(p), 4)
        self.assertEqual(p[ChebyshevVector({x: 10, y: 4})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({x: 10, y: 2})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({y: 4})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({y: 2})], 1 / 4)

        # Multiplication by wrong type.
        c = ChebyshevVector({x: 3, y: 4})
        m = MonomialVector({x: 5, y: 2})
        with self.assertRaises(TypeError):
            c * m

    def test_derivative(self):

        # Derivative of 1 is 0.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        c = ChebyshevVector({x: 5, y: 4})
        p = c.derivative(z)
        self.assertEqual(len(p), 0)
        self.assertEqual(p[ChebyshevVector({})], 0)

        # Derivative of odd power.
        p = c.derivative(x)
        self.assertEqual(len(p), 3)
        self.assertEqual(p[ChebyshevVector({y: 4})], 5)
        self.assertEqual(p[ChebyshevVector({x: 2, y: 4})], 10)
        self.assertEqual(p[ChebyshevVector({x: 4, y: 4})], 10)

        # Derivative of even power.
        p = c.derivative(y)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[ChebyshevVector({x: 5, y: 1})], 8)
        self.assertEqual(p[ChebyshevVector({x: 5, y: 3})], 8)

    def test_integral(self):

        # Power 0.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        c = ChebyshevVector({y: 1, z: 3})
        p = c.integral(x)
        self.assertEqual(len(p), 1)
        self.assertEqual(p[ChebyshevVector({x: 1, y: 1, z: 3})], 1)

        # Power 1.
        p = c.integral(y)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[ChebyshevVector({z: 3})], 1 / 4)
        self.assertEqual(p[ChebyshevVector({y: 2, z: 3})], 1 / 4)

        # Power > 1.
        p = c.integral(z)
        self.assertEqual(len(p), 2)
        self.assertEqual(p[ChebyshevVector({y: 1, z: 2})], - 1 / 4)
        self.assertEqual(p[ChebyshevVector({y: 1, z: 4})], 1 / 8)

    def test_in_monomial_basis(self):

        # Zero-dimensional.
        m = ChebyshevVector({})
        p = Polynomial({MonomialVector({}): 1})
        self.assertEqual(m.in_monomial_basis(), p)

        # One-dimensional.
        x = Variable('x')
        m = ChebyshevVector({x: 9})
        p = Polynomial({
            MonomialVector({x: 1}): 9,
            MonomialVector({x: 3}): - 120,
            MonomialVector({x: 5}): 432,
            MonomialVector({x: 7}): - 576,
            MonomialVector({x: 9}): 256,
            })
        self.assertEqual(m.in_monomial_basis(), p)

        # Two-dimensional.
        y = Variable('y')
        m = ChebyshevVector({x: 4, y: 3})
        p = Polynomial({
            MonomialVector({y: 1}): - 3,
            MonomialVector({y: 3}): 4,
            MonomialVector({x: 2, y: 1}): 24,
            MonomialVector({x: 2, y: 3}): - 32,
            MonomialVector({x: 4, y: 1}): - 24,
            MonomialVector({x: 4, y: 3}): 32,
            })
        self.assertEqual(m.in_monomial_basis(), p)

    def test_repr(self):

        x = Variable('x')
        x3 = Variable('x', 3)
        c = ChebyshevVector({x: 5, x3: 2})
        self.assertEqual(c.__repr__(), 'T_{5}(x)T_{2}(x_{3})')
