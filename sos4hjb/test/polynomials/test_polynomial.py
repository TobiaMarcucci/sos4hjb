import unittest

from sos4hjb.polynomials import (Variable, BasisVector, MonomialVector,
                                          ChebyshevVector, Polynomial)

class TestPolynomial(unittest.TestCase):

    def test_init(self):

        # Wrong initialization.
        x0 = Variable('x', 0)
        x1 = Variable('x', 1)
        x2 = Variable('x', 2)
        m0 = MonomialVector({x0: 4, x1: 1})
        m1 = MonomialVector({x0: 4, x2: 2})
        m2 = MonomialVector({x1: 0, x2: 11})
        with self.assertRaises(ValueError):
            p = Polynomial([1, 2, 3, 4], [m0, m1, m2])
        c2 = ChebyshevVector({x1: 0, x2: 11})
        with self.assertRaises(ValueError):
            p = Polynomial([1, 2, 3], [m0, m1, c2])
        with self.assertRaises(ValueError):
            p = Polynomial([1, 2, '3'], [m0, m1, m2])
        with self.assertRaises(ValueError):
            p = Polynomial([1, 2, '3'], [m0, m1, x2])

        # Ordering.
        p = Polynomial([1, 2, 3], [m0, m1, m2])
        self.assertEqual(p.basis, [m2, m1, m0])
        self.assertEqual(p.coefficients, [3, 2, 1])

        # Remove duplicates.
        p = Polynomial([1, 2, 3, 4], [m0, m1, m2, m1])
        self.assertEqual(p.basis, [m2, m1, m0])
        self.assertEqual(p.coefficients, [3, 6, 1])
