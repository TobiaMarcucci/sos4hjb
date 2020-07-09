import unittest

from sos4hjb.polynomials import (Variable, BasisVector, MonomialVector,
                                          ChebyshevVector, Polynomial)

class TestPolynomial(unittest.TestCase):

    def test_init(self):

        for vector_type in (MonomialVector, ChebyshevVector):

            # Empty initialization.
            p = Polynomial({})
            self.assertEqual(p.coef_dict, {})

            # Simple initialization.
            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 4, z: 2})
            coef_dict = {v0: 2, v1: 3.22}
            p = Polynomial(coef_dict)
            self.assertEqual(p.coef_dict, coef_dict)

            # Removes zeros.
            coef_dict[v1] = 0
            p = Polynomial(coef_dict)
            self.assertEqual(p.coef_dict, {v0: 2})

            # Non-vector in the vectors.
            with self.assertRaises(TypeError):
                Polynomial({v0: 2, 2: 3.22})
            with self.assertRaises(TypeError):
                Polynomial({v0: 2, x: 3.22})

        # Vector of different types.
        m = MonomialVector({x: 4, y: 1})
        c = ChebyshevVector({x: 4, z: 2})
        with self.assertRaises(TypeError):
            Polynomial({m: 2, c: 3.22})

    def test_getter_setter(self):

        for vector_type in (MonomialVector, ChebyshevVector):

            # Getter.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 5, y: 12})
            p = Polynomial({v0: 2, v1: 3.22})
            self.assertEqual(p[v0], 2)
            self.assertEqual(p[v1], 3.22)
            self.assertEqual(p[v2], 0)
            self.assertEqual(len(p), 2)

            # Setter.
            p[v0] = 1.11
            p[v2] = 6
            self.assertEqual(p[v0], 1.11)
            self.assertEqual(p[v1], 3.22)
            self.assertEqual(p[v2], 6)
            self.assertEqual(len(p), 3)

            # Delete instead of setting to zero.
            p[v2] = 0
            self.assertEqual(p[v2], 0)
            self.assertEqual(len(p), 2)

            # Do not set if zero.
            p[v2] = 0
            self.assertEqual(p[v2], 0)

            # Non-vector in the vectors.
            with self.assertRaises(TypeError):
                p[2] = 5
            with self.assertRaises(TypeError):
                p[x] = 5

        # Vector of different types.
        m = MonomialVector({x: 4, y: 1})
        c = ChebyshevVector({x: 4, y: 2})
        p = Polynomial({m: 2.1})
        with self.assertRaises(TypeError):
            p[c] = 5
