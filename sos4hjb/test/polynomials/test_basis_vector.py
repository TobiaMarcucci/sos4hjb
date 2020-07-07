import unittest

from sos4hjb.polynomials import Variable, BasisVector

class TestBasisVector(unittest.TestCase):

    def test_init(self):

        # Correct initializations.
        v = BasisVector({})
        self.assertEqual(v.power_dict, {})
        self.assertEqual(v.degree, 0)
        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        power_dict = {x0: 5, x1: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, power_dict)
        self.assertEqual(v.degree, 7)

        # Removes zeros.
        x2 = Variable('x', 2)
        power_dict[x2] = 0
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, {x0: 5, x1: 2})
        self.assertEqual(v.degree, 7)

        # Wrong initializations.
        with self.assertRaises(ValueError):
            BasisVector([5, 2])
        with self.assertRaises(ValueError):
            BasisVector({'x_{0}': 5, x1: 2})
        with self.assertRaises(ValueError):
            BasisVector({x0: 5.5, x1: 2})
        with self.assertRaises(ValueError):
            BasisVector({x0: 5.5, x1: -1})


    def test_getter_and_setter(self):

        # Getter.
        x0 = Variable('x', 0)
        x3 = Variable('x', 3)
        x7 = Variable('x', 7)
        power_dict = {x0: 5, x3: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v[x0], 5)
        self.assertEqual(v[x3], 2)
        self.assertEqual(v[x7], 0)
        with self.assertRaises(ValueError):
            v['x_{7}']

        # Setter.
        x9 = Variable('x', 9)
        v[x0] = 2
        v[x9] = 6
        self.assertEqual(v[x0], 2)
        self.assertEqual(v[x9], 6)
        with self.assertRaises(ValueError):
            v['x_{0}'] = 2
        with self.assertRaises(ValueError):
            v[x0] = 2.5
        with self.assertRaises(ValueError):
            v[x0] = -1

    def test_orderings(self):

        # Basis vectors with the same variables.
        x0 = Variable('x', 0)
        x3 = Variable('x', 3)
        x7 = Variable('x', 7)
        v = BasisVector({x0: 5, x3: 2, x7: 3})
        w = BasisVector({x0: 5, x3: 4, x7: 3})
        self._test_all_orderings(v, w)

        # Basis vectors with different variables.
        v = BasisVector({x0: 5, x7: 1})
        w = BasisVector({x0: 5, x3: 2})
        self._test_all_orderings(v, w)

        # Basis vectors with different variable names.
        a0 = Variable('a', 0)
        a3 = Variable('a', 3)
        v = BasisVector({x0: 5, x3: 2})
        w = BasisVector({a0: 4, a3: 1})
        self._test_all_orderings(v, w)

    def _test_all_orderings(self, v, w):

        # Test all the methods for v < w.
        self.assertTrue(v < w)
        self.assertTrue(v <= w)
        self.assertTrue(w > v)
        self.assertTrue(w >= v)
        self.assertTrue(v != w)
        self.assertTrue(v == v)
        self.assertFalse(w < v)
        self.assertFalse(w <= v)
        self.assertFalse(v > w)
        self.assertFalse(v >= w)
        self.assertFalse(v != v)
        self.assertFalse(v == w)

    def test_representation(self):
        x0 = Variable('x', 0)
        x3 = Variable('x', 3)
        v = BasisVector({x0: 5, x3: 2})
        self.assertEqual(v.__repr__(), '(x_{0}, 5) (x_{3}, 2) ')

    def test_misc(self):

        # Degree and length.
        v = BasisVector({})
        self.assertEqual(v.degree, 0)
        self.assertEqual(len(v), 0)
        x0 = Variable('x', 0)
        x3 = Variable('x', 3)
        power_dict = {x0: 5, x3: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v.degree, 7)
        self.assertEqual(len(v), 2)

        # Iteration.
        for v, p in v:
            self.assertEqual(p, power_dict[v])
