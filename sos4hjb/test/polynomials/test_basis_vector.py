import unittest

from sos4hjb.polynomials import Variable, BasisVector

class TestBasisVector(unittest.TestCase):

    def test_init(self):

        # Simple initializations.
        v = BasisVector({})
        self.assertEqual(v.power_dict, {})
        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, power_dict)
        self.assertEqual(v.degree, 7)

        # Removes zeros.
        z = Variable('z')
        power_dict[z] = 0
        v = BasisVector(power_dict)
        self.assertEqual(v.power_dict, {x: 5, y: 2})
        self.assertEqual(v.degree, 7)

    def test_getter_and_setter(self):

        # Getter.
        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        power_dict = {x: 5, y: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v[x], 5)
        self.assertEqual(v[y], 2)
        self.assertEqual(v[z], 0)

        # Setter.
        v[x] = 2
        v[z] = 6
        self.assertEqual(v[x], 2)
        self.assertEqual(v[z], 6)

    def test_eq_ineq(self):

        x = Variable('x')
        y = Variable('y')
        z = Variable('z')
        u = BasisVector({x: 5, y: 2, z: 3})
        v = BasisVector({x: 5, y: 2, z: 3})
        w = BasisVector({x: 5, z: 3})
        self.assertTrue(u == v)
        self.assertTrue(u != w)
        self.assertTrue(v != w)
        
    def test_repr(self):
        
        x = Variable('x')
        x3 = Variable('x', 3)
        v = BasisVector({x: 5, x3: 2})
        self.assertEqual(v.__repr__(), '(x,5)(x_{3},2)')

    def test_misc(self):

        # Degree and length.
        v = BasisVector({})
        self.assertEqual(v.degree, 0)
        self.assertEqual(len(v), 0)
        x = Variable('x')
        y = Variable('y')
        power_dict = {x: 5, y: 2}
        v = BasisVector(power_dict)
        self.assertEqual(v.degree, 7)
        self.assertEqual(len(v), 2)

        # Iteration.
        for v, p in v:
            self.assertEqual(p, power_dict[v])
