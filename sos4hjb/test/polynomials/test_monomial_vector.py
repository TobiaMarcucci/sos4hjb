import unittest

from sos4hjb.polynomials import (Variable, MonomialVector, ChebyshevVector,
                                 Polynomial)

class TestMonomialVector(unittest.TestCase):

    def test_call(self):

        # Partial evaluation.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        power_dict = {x: 5, y: 2, z: 3}
        m = MonomialVector(power_dict)
        x_eval = {x: - 2.1, y: 1.5}
        m_eval = MonomialVector({z: 3})
        c_eval = (- 2.1) ** 5 * 1.5 ** 2
        p = Polynomial({m_eval: c_eval})
        self.assertAlmostEqual(m(x_eval), p)

        # Complete evaluation.
        x_eval[z] = 5
        m_eval = MonomialVector({})
        c_eval *= 5 ** 3
        p = Polynomial({m_eval: c_eval})
        self.assertAlmostEqual(m(x_eval), p)

        # Cancellation.
        x_eval = {z: 0}
        p = Polynomial({})
        self.assertAlmostEqual(m(x_eval), p)

    def test_mul(self):

        x = Variable('x')
        y = Variable('y')
        m0 = MonomialVector({x: 5, y: 2})
        z = Variable('z')
        m1 = MonomialVector({z: 3, y: 2})
        p = m0 * m1
        m01 = MonomialVector({x: 5, y: 4, z: 3})
        self.assertEqual(p[m01], 1)

        # Multiplication by wrong type.
        m = MonomialVector({x: 5, y: 2})
        c = ChebyshevVector({x: 3, y: 4})
        with self.assertRaises(TypeError):
            m * c

    def test_derivative(self):

        x = Variable('x')
        y = Variable('y')
        m = MonomialVector({x: 5, y: 2})
        p = m.derivative(y)
        m_der = MonomialVector({x: 5, y: 1})
        self.assertEqual(p[m_der], 2)

    def test_integral(self):

        x = Variable('x')
        y = Variable('y')
        m = MonomialVector({x: 5, y: 2})
        p = m.integral(y)
        m_pr = MonomialVector({x: 5, y: 3})
        self.assertEqual(p[m_pr], 1 / 3)

    def test_repr(self):

        x = Variable('x')
        x3 = Variable('x', 3)
        m = MonomialVector({x: 5, x3: 2})
        self.assertEqual(m.__repr__(), 'x^{5}x_{3}^{2}')

        # Suppress power equal to 1.
        m = MonomialVector({x: 5, x3: 1})
        self.assertEqual(m.__repr__(), 'x^{5}x_{3}')
