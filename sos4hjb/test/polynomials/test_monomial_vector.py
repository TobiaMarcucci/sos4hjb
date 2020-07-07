import unittest

from sos4hjb.polynomials import Variable, MonomialVector

class TestMonomialVector(unittest.TestCase):

    def test_call(self):

        # Legitimate evaluation dict.
        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        power_dict = {x0: 5, x1: 2}
        m = MonomialVector(power_dict)
        x_eval = {x0: 2.1, x1: 1.5}
        value = 2.1 ** 5 * 1.5 ** 2
        self.assertEqual(m(x_eval), value)
        x2 = Variable('x', 2)
        x_eval[x2] = 15
        self.assertEqual(m(x_eval), value)

        # Illegitimate evaluation dict.
        with self.assertRaises(ValueError):
            m({'x_{0}': 2.1, x1: 1.5})
        with self.assertRaises(ValueError):
            m([2.1, 1.5])
        with self.assertRaises(ValueError):
            m({x0: 2.1})

    def test_mul(self):

        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        m0 = MonomialVector({x0: 5, x1: 2})
        x2 = Variable('x', 2)
        m1 = MonomialVector({x2: 3, x1: 2})
        p = m0 * m1
        m01 = MonomialVector({x0: 5, x1: 4, x2: 3})
        self.assertEqual(p.coefficients, [1])
        self.assertEqual(p.basis, [m01])

    def test_derivative_and_primitive(self):

        # Derivative.
        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        m = MonomialVector({x0: 5, x1: 2})
        p = m.derivative(x1)
        m_der = MonomialVector({x0: 5, x1: 1})
        self.assertEqual(p.coefficients, [2])
        self.assertEqual(p.basis, [m_der])

        # Primitive.
        p = m.primitive(x1)
        m_pr = MonomialVector({x0: 5, x1: 3})
        self.assertEqual(p.coefficients, [1 / 3])
        self.assertEqual(p.basis, [m_pr])
