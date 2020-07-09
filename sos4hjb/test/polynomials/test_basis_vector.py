import unittest

from sos4hjb.polynomials import Variable, BasisVector

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
        u = BasisVector({x: 5, y: 2, z: 3})
        v = BasisVector({x: 5, y: 2, z: 3})
        w = BasisVector({x: 5, z: 3})
        self.assertTrue(u == v)
        self.assertTrue(u != w)
        self.assertTrue(v != w)

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
        self.assertEqual(set(vector.variables), set(power_dict))
        self.assertEqual(set(vector.powers), set(power_dict.values()))

    def test_degree(self):

        v = BasisVector({})
        self.assertEqual(v.degree, 0)
        x = Variable('x')
        y = Variable('y')
        v = BasisVector({x: 5, y: 2})
        self.assertEqual(v.degree, 7)

    def test_repr(self):

        x = Variable('x')
        x3 = Variable('x', 3)
        v = BasisVector({x: 5, x3: 2})
        self.assertEqual(v.__repr__(), '(x,5)(x_{3},2)')
