import unittest
import numpy as np

from sos4hjb.polynomials import (Variable, BasisVector, MonomialVector,
                                          ChebyshevVector, Polynomial)

vector_types = (MonomialVector, ChebyshevVector)

class TestPolynomial(unittest.TestCase):

    def test_init(self):

        for vector_type in vector_types:

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

        for vector_type in vector_types:

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

    def test_call(self):

        # Monomial.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        p = Polynomial({
            MonomialVector({x: 2, y: 2}): 3.5,
            MonomialVector({x: 3, z: 5}): .5
            })
        eval_dict = {x: 2, y: .3, z: 1.5}
        value = 3.5 * (2 ** 2 * .3 ** 2)
        value += .5 * (2 ** 3 * 1.5 ** 5)
        self.assertEqual(p(eval_dict), value)

        # Chebyshev.
        p = Polynomial({
            ChebyshevVector({x: 2, y: 2}): 3.5,
            ChebyshevVector({x: 3, z: 5}): .5
            })
        T = ChebyshevVector._call_univariate
        value = 3.5 * (T(2, 2) * T(2, .3))
        value += .5 * (T(3, 2) * T(5, 1.5))
        self.assertEqual(p(eval_dict), value)

    def test_eq_ne(self):

        for vector_type in vector_types:

            # Comparisons with different lengths.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 6})
            p1 = Polynomial({v0: 5, v1: 2.5, v2: 3})
            p2 = Polynomial({v1: 2.5, v0: 5, v2: 3})
            p3 = Polynomial({v0: 5, v2: 3})
            p4 = Polynomial({v0: 5, v1: 0, v2: 3})
            self.assertTrue(p1 == p2)
            self.assertTrue(p1 != p3)
            self.assertTrue(p2 != p3)
            self.assertTrue(p3 == p4)

            # Comparison to 0.
            self.assertFalse(p1 == 0)
            self.assertTrue(p1 != 0)
            p = Polynomial({})
            self.assertTrue(p == 0)
            self.assertFalse(p != 0)

        # Comparison with different vector type.
        m = MonomialVector({x: 4, y: 1})
        c = ChebyshevVector({x: 4, y: 1})
        pm = Polynomial({m: 1})
        pc = Polynomial({c: 1})
        self.assertTrue(pm != pc)

    def test_len(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 6})
            p = Polynomial({v0: 5, v1: 2.5, v2: 3})
            self.assertEqual(len(p), 3)

    def test_iter(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, z: 2})
            v2 = vector_type({x: 6})
            coef_dict = {v0: 5, v1: 2.5, v2: 3}
            p = Polynomial(coef_dict)
            for v, c in p:
                self.assertEqual(c, coef_dict[v])
            self.assertEqual(set(p.vectors), set(coef_dict))
            self.assertEqual(set(p.coefficients), set(coef_dict.values()))
            self.assertEqual(set(p.variables), set([x, y, z]))

    def test_add_sub(self):

        for vector_type in vector_types:

            # Addition.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 6})
            p0 = Polynomial({v1: 2.5, v2: 3})
            p1 = Polynomial({v1: 2, v0: 3.33})
            p01 = Polynomial({v0: 3.33, v1: 4.5, v2: 3})
            self.assertEqual(p0 + p1, p01)

            # Iterative addition.
            p0 += p1
            self.assertEqual(p0, p01)

            # Subtraction.
            p0 = Polynomial({v1: 2.5, v2: 3})
            p1 = Polynomial({v1: 2, v0: 3.33})
            p01 = Polynomial({v0: -3.33, v1: .5, v2: 3})
            self.assertEqual(p0 - p1, p01)

            # Iterative subtraction.
            p0 -= p1
            self.assertEqual(p0, p01)

    def test_mul(self):

        # Multiplication by polynomial (monomial).
        x = Variable('x')
        y = Variable('y')
        v0 = MonomialVector({x: 1, y: 3})
        v1 = MonomialVector({x: 2})
        v2 = MonomialVector({x: 2, y: 1})
        p0 = Polynomial({v0: 3, v1: 5})
        p1 = Polynomial({v2: 2})
        p01 = Polynomial({
            MonomialVector({x: 3, y: 4}): 6,
            MonomialVector({x: 4, y: 1}): 10,
            })
        self.assertEqual(p0 * p1, p01)

        # Iterative multiplication by polynomial (monomial).
        p0 *= p1
        self.assertEqual(p0, p01)


        # Multiplication by polynomial (Chebyshev).
        v0 = ChebyshevVector({x: 1, y: 3})
        v1 = ChebyshevVector({x: 2})
        v2 = ChebyshevVector({x: 2, y: 1})
        p0 = Polynomial({v0: 3, v1: 5})
        p1 = Polynomial({v2: 2})
        p01 = Polynomial({
            ChebyshevVector({x: 3, y: 4}): 1.5,
            ChebyshevVector({x: 3, y: 2}): 1.5,
            ChebyshevVector({x: 1, y: 4}): 1.5,
            ChebyshevVector({x: 1, y: 2}): 1.5,
            ChebyshevVector({x: 4, y: 1}): 5,
            ChebyshevVector({y: 1}): 5
            })
        self.assertEqual(p0 * p1, p01)

        # Iterative multiplication by polynomial (Chebyshev).
        p0 *= p1
        self.assertEqual(p0, p01)

        for vector_type in vector_types:

            # Multiplication by scalar.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 6})
            p = Polynomial({v1: 2.5, v2: 3})
            c = 6
            p6 = Polynomial({v1: 15, v2: 18})
            self.assertEqual(p * 6, p6)

            # Iterative multiplication by scalar.
            p *= 6
            self.assertEqual(p, p6)

    def test_pow(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 1, y: 3})
            v1 = vector_type({x: 2, y: 2})
            p = Polynomial({v0: 3, v1: 5})
            p0 = Polynomial({vector_type({}): 1})
            self.assertEqual(p ** 0, p0)
            p_pow = Polynomial({v0: 3, v1: 5})
            for i in range(1, 5):
                self.assertEqual(p ** i, p_pow)
                p_pow *= p

    def test_round(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 1, y: 3})
            v1 = vector_type({x: 2, y: 2})
            v2 = vector_type({x: 4, y: 1})
            p = Polynomial({v0: 1 / 3, v1: - 5.2, v2: .22233})
            p0 = Polynomial({v1: - 5})
            self.assertEqual(round(p), p0)
            p1 = Polynomial({v0: .3, v1: - 5.2, v2: .2})
            self.assertEqual(round(p, 1), p1)
            p4 = Polynomial({v0: .3333, v1: - 5.2, v2: .2223})
            self.assertEqual(round(p, 4), p4)

    def test_abs(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 1, y: 3})
            v1 = vector_type({x: 2, y: 2})
            v2 = vector_type({x: 4, y: 1})
            p = Polynomial({v0: 1 / 3, v1: - 5.2, v2: .22233})
            p_abs = Polynomial({v0: 1 / 3, v1: 5.2, v2: .22233})
            self.assertEqual(abs(p), p_abs)

    def test_derivative(self):
        
        # Monomial.
        x = Variable('x')
        y = Variable('y')
        m0 = MonomialVector({x: 4, y: 1})
        m1 = MonomialVector({x: 5, y: 2})
        p = Polynomial({m0: 2.5, m1: -3})
        px = Polynomial({
            MonomialVector({x: 3, y: 1}): 10,
            MonomialVector({x: 4, y: 2}): - 15
            })
        self.assertEqual(p.derivative(x), px)
        py = Polynomial({
            MonomialVector({x: 4}): 2.5,
            MonomialVector({x: 5, y: 1}): - 6
            })
        self.assertEqual(p.derivative(y), py)
        z = Variable('z')
        pz = Polynomial({})
        self.assertEqual(p.derivative(z), pz)

        # Chebyshev.
        x = Variable('x')
        y = Variable('y')
        c0 = ChebyshevVector({x: 4, y: 1})
        c1 = ChebyshevVector({x: 5, y: 2})
        p = Polynomial({c0: - 2.5, c1: 3})
        px = Polynomial({
            ChebyshevVector({x: 1, y: 1}): - 20,
            ChebyshevVector({x: 3, y: 1}): - 20,
            ChebyshevVector({y: 2}): 15,
            ChebyshevVector({x: 2, y: 2}): 30,
            ChebyshevVector({x: 4, y: 2}): 30,
            })
        self.assertEqual(p.derivative(x), px)
        py = Polynomial({
            ChebyshevVector({x: 4}): - 2.5,
            ChebyshevVector({x: 5, y: 1}): 12,
            })
        self.assertEqual(p.derivative(y), py)
        z = Variable('z')
        pz = Polynomial({})
        self.assertEqual(p.derivative(z), pz)

    def test_primitive(self):

        # Monomial.
        x = Variable('x')
        y = Variable('y')
        m0 = MonomialVector({x: 4, y: 1})
        m1 = MonomialVector({x: 5, y: 2})
        p = Polynomial({m0: - 2.5, m1: 3})
        px = Polynomial({
            MonomialVector({x: 5, y: 1}): - .5,
            MonomialVector({x: 6, y: 2}): .5
            })
        self.assertEqual(p.primitive(x), px)
        py = Polynomial({
            MonomialVector({x: 4, y: 2}): - 1.25,
            MonomialVector({x: 5, y: 3}): 1
            })
        self.assertEqual(p.primitive(y), py)
        z = Variable('z')
        pz = Polynomial({
            MonomialVector({x: 4, y: 1, z: 1}): - 2.5,
            MonomialVector({x: 5, y: 2, z: 1}): 3,
            })
        self.assertEqual(p.primitive(z), pz)

        # Chebyshev.
        x = Variable('x')
        y = Variable('y')
        c0 = ChebyshevVector({x: 2, y: 1})
        c1 = ChebyshevVector({x: 5, y: 2})
        p = Polynomial({c0: 3, c1: -6})
        px = Polynomial({
            ChebyshevVector({x: 3, y: 1}): .5,
            ChebyshevVector({x: 1, y: 1}): - 1.5,
            ChebyshevVector({x: 6, y: 2}): - .5,
            ChebyshevVector({x: 4, y: 2}): .75
            })
        self.assertEqual(p.primitive(x), px)
        py = Polynomial({
            ChebyshevVector({x: 2}): .75,
            ChebyshevVector({x: 2, y: 2}): .75,
            ChebyshevVector({x: 5, y: 3}): - 1,
            ChebyshevVector({x: 5, y: 1}): 3
            })
        self.assertEqual(p.primitive(y), py)
        z = Variable('z')
        pz = Polynomial({
            ChebyshevVector({x: 2, y: 1, z: 1}): 3,
            ChebyshevVector({x: 5, y: 2, z: 1}): - 6,
            })
        self.assertEqual(p.primitive(z), pz)

    def test_degree(self):
        
        for vector_type in vector_types:

            # Degrees 8 and 5.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 3})
            p = Polynomial({v0: 2.5, v1: 3})
            self.assertEqual(p.degree, 8)
            p[v1] = 0
            self.assertEqual(p.degree, 5)

            # Degree 0.
            v = vector_type({})
            p = Polynomial({v: 2.5})
            self.assertEqual(p.degree, 0)

            # Degree 0 for empty polynomial.
            p = Polynomial({})
            self.assertEqual(p.degree, 0)

    def test_is_odd_is_even(self):
        
        for vector_type in vector_types:

            # Even.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 2})
            v1 = vector_type({x: 1, y: 1})
            p = Polynomial({v0: 2.5, v1: 3})
            self.assertTrue(p.is_even)
            self.assertFalse(p.is_odd)

            # Not even nor odd.
            v0[y] += 1
            self.assertFalse(p.is_odd)
            self.assertFalse(p.is_even)

            # Odd.
            v1[y] += 1
            self.assertTrue(p.is_odd)
            self.assertFalse(p.is_even)

            # Degree 0.
            v = vector_type({})
            p = Polynomial({v: 1})
            self.assertTrue(p.is_even)
            self.assertFalse(p.is_odd)

            # Empty polynomial is 0, hence even.
            p = Polynomial({})
            self.assertTrue(p.is_even)
            self.assertFalse(p.is_odd)

    def test_quadratic_form(self):

        for vector_type in vector_types:

            x = Variable.multivariate('x', 2)
            basis = vector_type.construct_basis(x, 1)
            Q = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
            p = Polynomial.quadratic_form(basis, Q)
            p_target = basis[0] * basis[0] + (basis[0] * basis[1]) * 4 + \
                       (basis[0] * basis[2]) * 6 + (basis[1] * basis[1]) * 4 + \
                       (basis[1] * basis[2]) * 10 + (basis[2] * basis[2]) * 6
            self.assertEqual(p, p_target)

    def test_repr(self):

        for vector_type in vector_types:

            # Only + signs.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            p = Polynomial({v0: 2.5, v1: 3})
            r = '2.5' + v0.__repr__() + '+3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)

            # With - signs.
            p = Polynomial({v0: 2.5, v1: - 3})
            r = '2.5' + v0.__repr__() + '-3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            p = Polynomial({v0: - 2.5, v1: 3})
            r = '-2.5' + v0.__repr__() + '+3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)

            # 0 if all the coefficients are 0.
            p = Polynomial({})
            self.assertEqual(p.__repr__(), '0')

            # Suppress 1 coefficients.
            p = Polynomial({v0: 2.5, v1: 1})
            r = '2.5' + v0.__repr__() + '+' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            p = Polynomial({v0: 1, v1: 3.44})
            r = v0.__repr__() + '+3.44' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)

            # Vector with zero power.
            v2 = vector_type({})
            p = Polynomial({v2: 5.33, v0: 2, v1: 3})
            r = '5.33+2' + v0.__repr__() + '+3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
