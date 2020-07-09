import unittest

from sos4hjb.polynomials import Variable

class TestVariable(unittest.TestCase):

    def test_init(self):

        # Variable without index.
        x = Variable('x')
        self.assertEqual(x.name, 'x')
        self.assertEqual(x.index, 0)

        # Variable with index.
        x1 = Variable('x', 1)
        self.assertEqual(x1.name, 'x')
        self.assertEqual(x1.index, 1)

        # Wrong initializations.
        with self.assertRaises(TypeError):
            Variable(5)
        with self.assertRaises(ValueError):
            Variable('x', .5)
        with self.assertRaises(ValueError):
            Variable('x', -1)

    def test_eq_ineq(self):

        # Variable without index.
        x = Variable('x')
        y = Variable('x')
        self.assertTrue(x == y)
        self.assertFalse(x != y)
        y = Variable('y')
        self.assertFalse(x == y)
        self.assertTrue(x != y)

        # Variable with index.
        x1 = Variable('x', 1)
        x2 = Variable('x', 1)
        self.assertTrue(x1 == x2)
        self.assertFalse(x1 != x2)
        x2 = Variable('x', 2)
        self.assertFalse(x1 == x2)
        self.assertTrue(x1 != x2)

    def test_repr(self):

        # Variable without index.
        x = Variable('x')
        self.assertEqual(x.__repr__(), 'x')

        # Variable with index.
        x1 = Variable('x', 1)
        x11 = Variable('x', 11)
        self.assertEqual(x1.__repr__(), 'x_{1}')
        self.assertEqual(x11.__repr__(), 'x_{11}')

    def test_multivariate(self):

        # Test the representation only.
        x = Variable.multivariate('x', 5)
        for i, xi in enumerate(x):
            self.assertTrue(xi.__repr__() == f'x_{{{i+1}}}')
