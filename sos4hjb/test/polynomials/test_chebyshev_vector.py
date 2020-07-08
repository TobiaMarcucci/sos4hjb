import unittest

from sos4hjb.polynomials import Variable, ChebyshevVector

class TestChebyshevVector(unittest.TestCase):

    def test_call(self):

        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        c = ChebyshevVector(power_dict)
        x_eval = {x: 2.1, y: 1.5}
        value0 = 16 * (2.1 ** 5) - 20 * (2.1 ** 3) + 5 * 2.1
        value1 = 2 * (1.5 ** 2) - 1
        value = value0 * value1
        self.assertAlmostEqual(c(x_eval), value)
        z = Variable('z')
        x_eval[z] = 15
        self.assertAlmostEqual(c(x_eval), value)

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

    def test_primitive(self):

        pass

    def test_repr(self):
        
        x = Variable('x')
        x3 = Variable('x', 3)
        c = ChebyshevVector({x: 5, x3: 2})
        self.assertEqual(c.__repr__(), 'T_{5}(x)T_{2}(x_{3})')
