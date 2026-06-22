########################################################################################
##
##                                  TESTS FOR
##                               'blocks.bvp.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.blocks.bvp import BVP1D


# TESTS ================================================================================

class TestBVP1D(unittest.TestCase):
    """
    Test the implementation of the 'BVP1D' block class.
    """

    def test_init(self):
        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - 1.0])
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2), n_nodes=11)

        self.assertEqual(bvp.n, 2)
        self.assertEqual(bvp.domain, (0.0, np.pi/2))
        self.assertEqual(bvp.x_eval.size, 11)
        #output pre-sized to n * len(x_eval)
        self.assertEqual(len(bvp.outputs), 22)
        #no free parameters
        self.assertIsNone(bvp.parameters())


    def test_algebraic_path(self):
        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - 1.0])
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2))
        self.assertEqual(len(bvp), 1)
        bvp.off()
        self.assertEqual(len(bvp), 0)


    def test_solve_sine(self):
        #y'' = -y, y(0)=0, y(pi/2)=1  ->  y = sin(x)
        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - 1.0])

        xq = np.linspace(0.0, np.pi/2, 9)
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2), x_eval=xq)
        bvp.update(0.0)

        self.assertTrue(bvp.success)
        y = bvp.solution()
        np.testing.assert_allclose(y[0], np.sin(xq), atol=1e-5)


    def test_eigenvalue_free_parameter(self):
        #y'' + p^2 y = 0, y(0)=y(1)=0, y'(0)=1  ->  p -> pi
        def fun(x, y, p, u):
            return np.vstack([y[1], -p[0]**2 * y[0]])

        def bc(ya, yb, p, u):
            return np.array([ya[0], yb[0], ya[1] - 1.0])

        y0 = lambda x: np.vstack([np.sin(np.pi*x), np.pi*np.cos(np.pi*x)])
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, 1.0), p0=[3.0], y0=y0)
        bvp.update(0.0)

        self.assertTrue(bvp.success)
        self.assertAlmostEqual(float(bvp.parameters()[0]), np.pi, places=5)


    def test_input_dependent_boundary(self):
        #y'' = -y, y(0)=0, y(pi/2)=u  ->  y = u*sin(x)
        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - u[0]])

        xq = np.linspace(0.0, np.pi/2, 9)
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2), x_eval=xq)

        bvp.inputs[0] = 3.0
        bvp.update(0.0)

        self.assertTrue(bvp.success)
        np.testing.assert_allclose(bvp.solution()[0], 3.0*np.sin(xq), atol=1e-5)


    def test_warmstart_and_reset(self):
        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - 1.0])
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2))

        bvp.update(0.0)
        self.assertTrue(bvp.success)
        #mesh was refined / updated by the solve
        self.assertGreaterEqual(bvp.x.size, 2)

        bvp.reset()
        self.assertFalse(bvp.success)
        np.testing.assert_array_equal(bvp.x, bvp._x0)


    def test_info(self):
        info = BVP1D.info()
        self.assertEqual(info["type"], "BVP1D")
        for p in ("fun", "bc", "n", "domain", "n_nodes", "x_eval", "y0", "p0", "tol"):
            self.assertIn(p, info["parameters"])


    def test_simulation(self):
        #drive the right boundary with a constant source, output tracks u*sin(x)
        from pathsim import Simulation, Connection
        from pathsim.blocks import Constant, Scope

        fun = lambda x, y, p, u: np.vstack([y[1], -y[0]])
        bc = lambda ya, yb, p, u: np.array([ya[0], yb[0] - u[0]])

        xq = np.linspace(0.0, np.pi/2, 5)
        src = Constant(2.0)
        bvp = BVP1D(fun, bc, n=2, domain=(0.0, np.pi/2), x_eval=xq)
        sco = Scope()

        sim = Simulation(
            blocks=[src, bvp, sco],
            connections=[Connection(src, bvp), Connection(bvp[0], sco[0])],
            log=False
            )
        sim.run(0.1)

        #midpoint value y(pi/4) = 2*sin(pi/4)
        np.testing.assert_allclose(bvp.solution()[0, 2], 2.0*np.sin(np.pi/4), atol=1e-5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
