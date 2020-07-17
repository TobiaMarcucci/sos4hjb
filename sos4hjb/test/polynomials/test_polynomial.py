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

        for vector_type in vector_types:

            # Partial evaluation.
            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            v0 = vector_type({x: 1, y: 2})
            v1 = vector_type({x: 3, z: 5})
            p = Polynomial({v0: 3.5, v1: .5})
            eval_dict = {x: 2, y: .3, z: - 3.12}
            value = v0(eval_dict) * 3.5 + v1(eval_dict) * .5
            self.assertAlmostEqual(p(eval_dict), value)

    def test_substitute(self):

        for vector_type in vector_types:

            # Partial evaluation.
            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            v0 = vector_type({x: 1, y: 2})
            v1 = vector_type({x: 3, z: 5})
            p = Polynomial({v0: 3.5, v1: .5})
            eval_dict = {x: 2, y: .3}
            p_eval = v0.substitute(eval_dict) * 3.5 + v1.substitute(eval_dict) * .5
            self.assertAlmostEqual(p.substitute(eval_dict), p_eval)

            # Complete evaluation.
            eval_dict[z] = - 3.12
            p_eval = v0.substitute(eval_dict) * 3.5 + v1.substitute(eval_dict) * .5
            self.assertAlmostEqual(p.substitute(eval_dict), p_eval)

            # Cancellation.
            eval_dict = {x: 0}
            p_eval = Polynomial({})
            self.assertAlmostEqual(p.substitute(eval_dict), p_eval)

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
            self.assertEqual(set(p.vectors()), set(coef_dict))
            self.assertEqual(set(p.coefficients()), set(coef_dict.values()))
            self.assertEqual(set(p.variables()), set([x, y, z]))

    def test_add_iadd_sub_isub(self):

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

    def test_mul_imul_rmul(self):

        for vector_type in vector_types:

            # Multiplication by scalar.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 2})
            v2 = vector_type({x: 6})
            p = Polynomial({v1: 2.5, v2: 3})
            p6 = Polynomial({v1: 15, v2: 18})
            self.assertEqual(p * 6, p6)
            self.assertEqual(6 * p, p6)
            p0 = Polynomial({})
            self.assertEqual(p * 0, p0)
            self.assertEqual(0 * p, p0)

            # Iterative multiplication by scalar.
            p *= 6
            self.assertEqual(p, p6)

            # Multiplication by polynomial.
            p0 = Polynomial({v0: 3.1, v1: 5.5})
            p1 = Polynomial({v0: -2, v2: 2.9})
            p01 = (v0 * v0) * 3.1 * (-2) + \
                  (v0 * v2) * 3.1 * 2.9 + \
                  (v1 * v0) * 5.5 * (-2) + \
                  (v1 * v2) * 5.5 * 2.9
            self.assertEqual(p0 * p1, p01)

            # Iterative multiplication by polynomial.
            p0 *= p1
            self.assertEqual(p0, p01)

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

        # 0 ** 0 is undefined.
        p = Polynomial({})
        with self.assertRaises(ValueError):
            p ** 0

    def test_pos_neg(self):

        for vector_type in vector_types:

            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 1, y: 3})
            v1 = vector_type({x: 2, y: 2})
            v2 = vector_type({x: 4, y: 1})
            p = Polynomial({v0: 1 / 3, v1: - 5.2, v2: .22})
            q = Polynomial({v0: - 1 / 3, v1: 5.2, v2: - .22})
            p_pos = + p
            p_neg = - p
            self.assertEqual(p_pos, p)
            self.assertEqual(p_neg, q)

            # Positive and negative must return a copy.
            p_pos[v0] = 3
            p_neg[v0] = 3
            self.assertTrue(p_pos != p)
            self.assertTrue(p_neg != q)

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

    def test_derivative_jacobian(self):

        for vector_type in vector_types:
        
            # Derivative.
            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            m0 = vector_type({x: 4, y: 1})
            m1 = vector_type({x: 5, y: 2})
            p = Polynomial({m0: 2.5, m1: -3})
            px = m0.derivative(x) * 2.5 + m1.derivative(x) * (-3)
            py = m0.derivative(y) * 2.5 + m1.derivative(y) * (-3)
            pz = Polynomial({})
            self.assertEqual(p.derivative(x), px)
            self.assertEqual(p.derivative(y), py)
            self.assertEqual(p.derivative(z), pz)

            # Jacobian.
            for pi, qi in zip(p.jacobian([z, x, y]), np.array([pz, px, py])):
                self.assertEqual(pi, qi)

    def test_integral_definite_integral(self):

        for vector_type in vector_types:

            # Indefinite.
            x = Variable('x')
            y = Variable('y')
            z = Variable('z')
            m0 = vector_type({x: 4, y: 1})
            m1 = vector_type({x: 5, y: 2})
            p = Polynomial({m0: 2.5, m1: -3})
            px = m0.integral(x) * 2.5 + m1.integral(x) * (-3)
            py = m0.integral(y) * 2.5 + m1.integral(y) * (-3)
            pz = m0.integral(z) * 2.5 + m1.integral(z) * (-3)
            self.assertEqual(p.integral(x), px)
            self.assertEqual(p.integral(y), py)
            self.assertEqual(p.integral(z), pz)

            # Definite.
            lbs = [-3, -2, 2.12]
            ubs = [-1, 4, 5]
            px = px.substitute({x: -1}) - px.substitute({x: -3})
            self.assertEqual(p.definite_integral([x], lbs[:1], ubs[:1]), px)
            pxy = px.integral(y)
            pxy = pxy.substitute({y: 4}) - pxy.substitute({y: -2})
            self.assertEqual(p.definite_integral([x, y], lbs[:2], ubs[:2]), pxy)
            pxyz = pxy.integral(z)
            pxyz = pxyz.substitute({z: 5}) - pxyz.substitute({z: 2.12})
            self.assertEqual(p.definite_integral([x, y, z], lbs, ubs), pxyz)

            # Definite wrong lengths.
            with self.assertRaises(ValueError):
                p.definite_integral([x, y], lbs, ubs)
            with self.assertRaises(ValueError):
                p.definite_integral([x, y, z], lbs[:2], ubs)
            with self.assertRaises(ValueError):
                p.definite_integral([x, y, z], lbs, ubs[:2])

    def test_degree(self):
        
        for vector_type in vector_types:

            # Degrees 8 and 5.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 1})
            v1 = vector_type({x: 5, y: 3})
            p = Polynomial({v0: 2.5, v1: 3})
            self.assertEqual(p.degree(), 8)
            p[v1] = 0
            self.assertEqual(p.degree(), 5)

            # Degree 0.
            v = vector_type({})
            p = Polynomial({v: 2.5})
            self.assertEqual(p.degree(), 0)

            # Degree 0 for empty polynomial.
            p = Polynomial({})
            self.assertEqual(p.degree(), 0)

    def test_is_odd_is_even(self):
        
        for vector_type in vector_types:

            # Even.
            x = Variable('x')
            y = Variable('y')
            v0 = vector_type({x: 4, y: 2})
            v1 = vector_type({x: 1, y: 1})
            p = Polynomial({v0: 2.5, v1: 3})
            self.assertTrue(p.is_even())
            self.assertFalse(p.is_odd())

            # Not even nor odd.
            v0[y] += 1
            self.assertFalse(p.is_odd())
            self.assertFalse(p.is_even())

            # Odd.
            v1[y] += 1
            self.assertTrue(p.is_odd())
            self.assertFalse(p.is_even())

            # Degree 0.
            v = vector_type({})
            p = Polynomial({v: 1})
            self.assertTrue(p.is_even())
            self.assertFalse(p.is_odd())

            # Empty polynomial is 0, hence even.
            p = Polynomial({})
            self.assertTrue(p.is_even())
            self.assertFalse(p.is_odd())

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

    def test_in_chebyshev_basis(self):

        # Zero polynomial.
        p = Polynomial({})
        self.assertEqual(p.in_chebyshev_basis(), p)

        # Non-zero polynomial.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        m0 = MonomialVector({x: 4, y: 3})
        m1 = MonomialVector({x: 1, z: 3})
        p = Polynomial({m0: 4, m1: 3})
        p_cheb = m0.in_chebyshev_basis() * 4 + m1.in_chebyshev_basis() * 3
        self.assertEqual(p.in_chebyshev_basis(), p_cheb)

    def test_in_monomial_basis(self):

        # Zero polynomial.
        p = Polynomial({})
        self.assertEqual(p.in_monomial_basis(), p)

        # Non-zero polynomial.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        m0 = ChebyshevVector({x: 4, y: 3})
        m1 = ChebyshevVector({x: 1, z: 3})
        p = Polynomial({m0: 4, m1: 3})
        p_mon = m0.in_monomial_basis() * 4 + m1.in_monomial_basis() * 3
        self.assertEqual(p.in_monomial_basis(), p_mon)

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
            self.assertEqual(p._repr_latex_(), '$' + r + '$')

            # With - signs.
            p = Polynomial({v0: 2.5, v1: - 3})
            r = '2.5' + v0.__repr__() + '-3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            self.assertEqual(p._repr_latex_(), '$' + r + '$')
            p = Polynomial({v0: - 2.5, v1: 3})
            r = '-2.5' + v0.__repr__() + '+3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            self.assertEqual(p._repr_latex_(), '$' + r + '$')

            # 0 if all the coefficients are 0.
            p = Polynomial({})
            self.assertEqual(p.__repr__(), '0')
            self.assertEqual(p._repr_latex_(), '$0$')

            # Suppress 1 coefficients.
            p = Polynomial({v0: 2.5, v1: 1})
            r = '2.5' + v0.__repr__() + '+' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            self.assertEqual(p._repr_latex_(), '$' + r + '$')
            p = Polynomial({v0: 1, v1: 3.44})
            r = v0.__repr__() + '+3.44' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            self.assertEqual(p._repr_latex_(), '$' + r + '$')

            # Vector with zero power.
            v2 = vector_type({})
            p = Polynomial({v2: 5.33, v0: 2, v1: 3})
            r = '5.33+2' + v0.__repr__() + '+3' + v1.__repr__()
            self.assertEqual(p.__repr__(), r)
            self.assertEqual(p._repr_latex_(), '$' + r + '$')
