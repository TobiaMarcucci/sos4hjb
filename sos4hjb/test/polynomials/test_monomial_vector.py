import unittest

from sos4hjb.polynomials import Variable, MonomialVector

class TestMonomialVector(unittest.TestCase):

    def test_call(self):

        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        m = MonomialVector(power_dict)
        x_eval = {x: 2.1, y: 1.5}
        value = 2.1 ** 5 * 1.5 ** 2
        self.assertEqual(m(x_eval), value)
        z = Variable('z', 2)
        x_eval[z] = 15
        self.assertEqual(m(x_eval), value)

    def test_mul(self):

        x = Variable('x')
        y = Variable('y')
        m0 = MonomialVector({x: 5, y: 2})
        z = Variable('z')
        m1 = MonomialVector({z: 3, y: 2})
        p = m0 * m1
        m01 = MonomialVector({x: 5, y: 4, z: 3})
        self.assertEqual(p[m01], 1)

    def test_derivative(self):

        x = Variable('x')
        y = Variable('y')
        m = MonomialVector({x: 5, y: 2})
        p = m.derivative(y)
        m_der = MonomialVector({x: 5, y: 1})
        self.assertEqual(p[m_der], 2)

    def test_primitive(self):

        x = Variable('x')
        y = Variable('y')
        m = MonomialVector({x: 5, y: 2})
        p = m.primitive(y)
        m_pr = MonomialVector({x: 5, y: 3})
        self.assertEqual(p[m_pr], 1 / 3)

    def test_repr(self):
        
        x = Variable('x')
        x3 = Variable('x', 3)
        m = MonomialVector({x: 5, x3: 2})
        self.assertEqual(m.__repr__(), 'x^{5}x_{3}^{2}')
