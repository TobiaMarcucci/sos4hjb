import unittest

from sos4hjb.polynomials import (Variable, BasisVector, MonomialVector,
                                 ChebyshevVector, Polynomial)

Vectors = (MonomialVector, ChebyshevVector)

class TestBasisVector(unittest.TestCase):

    def test_init(self):

        # Empty initialization.
        v = BasisVector({})
        self.assertEqual(v.power_dict, {})

        # Simple initialization.
        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, power_dict)

        # Removes zeros.
        z = Variable('z')
        power_dict[z] = 0
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, {x: 5, y: 2})

        # Non-variable variable.
        with self.assertRaises(TypeError):
            BasisVector({x: 5, 'y': 2})
        with self.assertRaises(TypeError):
            BasisVector({x: 5, 4: 2})

        # Non-integer power.
        power_dict[z] = 3.33
        with self.assertRaises(ValueError):
            BasisVector(power_dict)

    def test_getter_setter(self):

        # Getter.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        v = BasisVector({x: 5, y: 2})
        self.assertEqual(v[x], 5)
        self.assertEqual(v[y], 2)
        self.assertEqual(v[z], 0)
        self.assertEqual(len(v), 2)
        with self.assertRaises(TypeError):
            v['a']

        # Setter.
        v[x] = 12
        v[z] = 6
        self.assertEqual(v[x], 12)
        self.assertEqual(v[y], 2)
        self.assertEqual(v[z], 6)
        self.assertEqual(len(v), 3)

        # Delete instead of setting to zero.
        v[z] = 0
        self.assertEqual(v[z], 0)
        self.assertEqual(len(v), 2)

        # Do not set if zero.
        v[z] = 0
        self.assertEqual(v[z], 0)

        # Non-variable variable.
        with self.assertRaises(TypeError):
            v['z'] = 5
        with self.assertRaises(TypeError):
            v[4] = 5

        # Non-integer power.
        with self.assertRaises(ValueError):
            v[z] = 1.5
        with self.assertRaises(ValueError):
            v[z] = - 2

    def test_eq_ne(self):

        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        v1 = BasisVector({x: 5, y: 2, z: 3})
        v2 = BasisVector({y: 2, x: 5, z: 3})
        v3 = BasisVector({x: 5, z: 3})
        v4 = BasisVector({x: 5, y: 0, z: 3})
        self.assertTrue(v1 == v2)
        self.assertTrue(v1 != v3)
        self.assertTrue(v2 != v3)
        self.assertTrue(v3 == v4)

    def test_len(self):

        v = BasisVector({})
        self.assertEqual(len(v), 0)
        x = Variable('x')
        y = Variable('y')
        v = BasisVector({x: 5, y: 2})
        self.assertEqual(len(v), 2)

    def test_iter(self):

        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        vector = BasisVector(power_dict)
        for v, p in vector:
            self.assertEqual(p, power_dict[v])
        self.assertEqual(set(vector.variables()), set(power_dict))
        self.assertEqual(set(vector.powers()), set(power_dict.values()))

    def test_degree_is_odd_is_even(self):

        # Odd.
        v = BasisVector({})
        self.assertEqual(v.degree(), 0)
        x = Variable('x')
        y = Variable('y')
        v = BasisVector({x: 5, y: 2})
        self.assertEqual(v.degree(), 7)
        self.assertTrue(v.is_odd())
        self.assertFalse(v.is_even())

        # Even.
        v[y] = 3
        self.assertEqual(v.degree(), 8)
        self.assertTrue(v.is_even())
        self.assertFalse(v.is_odd())

        # 0 is even.
        v = BasisVector({})
        self.assertEqual(v.degree(), 0)
        self.assertTrue(v.is_even())
        self.assertFalse(v.is_odd())

    def test_make_polynomial(self):

        for Vector in Vectors:

            # Make a polynomial out of a number.
            p = Vector.make_polynomial(1)
            self.assertEqual(p, Polynomial({Vector({}): 1}))
            p = Vector.make_polynomial(- 3.14)
            self.assertEqual(p, Polynomial({Vector({}): - 3.14}))

            # Make a polynomial out of a variable.
            x = Variable('x')
            p = Vector.make_polynomial(x)
            self.assertEqual(p, Polynomial({Vector({x: 1}): 1}))

            # Type error otherwise.
            with self.assertRaises(TypeError):
                p = Vector.make_polynomial('a')
            with self.assertRaises(TypeError):
                p = Vector.make_polynomial(MonomialVector({}))
            with self.assertRaises(TypeError):
                p = Vector.make_polynomial(Polynomial({}))

    def test_construct_basis(self):

        # 2 variables, 3rd degree. Even and odd.
        x = Variable.multivariate('x', 2)
        degree = 3
        basis = BasisVector.construct_basis(x, degree)
        basis_powers = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (3, 0),
                        (0, 3), (2, 1), (1, 2)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 2 variables, 3rd degree. Even only.
        basis = BasisVector.construct_basis(x, degree, odd=False)
        basis_powers = [(0, 0), (2, 0), (0, 2), (1, 1)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 2 variables, 3rd degree. Odd only.
        basis = BasisVector.construct_basis(x, degree, even=False)
        basis_powers = [(1, 0), (0, 1), (3, 0), (0, 3), (2, 1), (1, 2)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 3 variables, 2nd degree. Even and odd.
        x = Variable.multivariate('x', 3)
        degree = 2
        basis = BasisVector.construct_basis(x, degree)
        basis_powers = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (2, 0, 0),
                        (0, 2, 0), (0, 0, 2), (1, 1, 0), (0, 1, 1), (1, 0, 1)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 3 variables, 2nd degree. Even only.
        basis = BasisVector.construct_basis(x, degree, odd=False)
        basis_powers = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0),
                        (0, 1, 1), (1, 0, 1)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 3 variables, 2nd degree. Odd only.
        basis = BasisVector.construct_basis(x, degree, even=False)
        basis_powers = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        self._test_basis_by_powers(x, basis, basis_powers)

        # 0 degree.
        degree = 0
        basis = BasisVector.construct_basis(x, degree)
        basis_powers = [(0, 0, 0)]
        self._test_basis_by_powers(x, basis, basis_powers)
        basis = BasisVector.construct_basis(x, degree, odd=False)
        self._test_basis_by_powers(x, basis, basis_powers)
        basis = BasisVector.construct_basis(x, degree, even=False)
        basis_powers = []
        self._test_basis_by_powers(x, basis, basis_powers)

        # Negative degree.
        with self.assertRaises(ValueError):
            basis = BasisVector.construct_basis(x, -3)

        # Float degree.
        with self.assertRaises(ValueError):
            basis = BasisVector.construct_basis(x, 3.5)

    def _test_basis_by_powers(self, variables, basis, basis_powers):
        self.assertEqual(len(basis), len(basis_powers))
        for powers in basis_powers:
            v = BasisVector(dict(zip(variables, powers)))
            self.assertTrue(v in basis)

    def test_repr(self):

        x = Variable('x')
        x3 = Variable('x', 3)
        v = BasisVector({x: 5, x3: 2})
        self.assertEqual(v.__repr__(), '(x,5)(x_{3},2)')
        self.assertEqual(v._repr_latex_(), '$(x,5)(x_{3},2)$')

        # 1 if all the powers are zero.
        v = BasisVector({})
        self.assertEqual(v.__repr__(), '1')
        self.assertEqual(v._repr_latex_(), '$1$')
