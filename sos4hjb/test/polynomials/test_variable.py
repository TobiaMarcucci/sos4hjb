import unittest

from sos4hjb.polynomials.variable import Variable

class TestVariable(unittest.TestCase):

    def test_initialization_and_representation(self):

        # Correct initializations.
        x = Variable('x')
        self.assertEqual(x.__repr__(), 'x')
        x = Variable('x', None)
        self.assertEqual(x.__repr__(), 'x')
        x0 = Variable('x', 0)
        self.assertEqual(x0.__repr__(), 'x_{0}')

        # Wrong initializations.
        with self.assertRaises(ValueError):
            Variable(0)
        with self.assertRaises(ValueError):
            Variable('x', 'a')
        with self.assertRaises(ValueError):
            Variable('x', -1)

    def test_orderings(self):

        # Different name.
        a = Variable('a')
        b = Variable('b')
        self._test_all_orderings(a, b)
        aa = Variable('aa')
        self._test_all_orderings(a, aa)

        # Different index.
        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        self._test_all_orderings(x0, x1)
        x3 = Variable('x', 3)
        x11 = Variable('x', 11)
        self._test_all_orderings(x3, x11)

        # Different index and name.
        blue3 = Variable('blue', 3)
        red0 = Variable('red', 0)
        self._test_all_orderings(blue3, red0)

        # With index vs without index.
        x = Variable('x')
        x0 = Variable('x', 0)
        self._test_all_orderings(x, x0)

    def _test_all_orderings(self, a, b):

        # Test all the methods for a < b.
        self.assertTrue(a < b)
        self.assertTrue(a <= b)
        self.assertTrue(b > a)
        self.assertTrue(b >= a)
        self.assertTrue(a != b)
        self.assertTrue(a == a)
        self.assertFalse(b < a)
        self.assertFalse(b <= a)
        self.assertFalse(a > b)
        self.assertFalse(a >= b)
        self.assertFalse(a != a)
        self.assertFalse(a == b)

    def test_multivariate(self):

        # Correct initializations.
        x = Variable.multivariate('x', 3)
        for i, xi in enumerate(x):
            self.assertTrue(xi.__repr__() == f'x_{{{i}}}')
