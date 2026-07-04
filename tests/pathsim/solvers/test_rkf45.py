########################################################################################
##
##                                  TESTS FOR 
##                             'solvers/rkf45.py'
##
##                              Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np

from pathsim.solvers.rkf45 import RKF45

from tests.pathsim.solvers._referenceproblems import PROBLEMS

import matplotlib.pyplot as plt


# TESTS ================================================================================

class TestRKF45(unittest.TestCase):
    """
    Test the implementation of the 'RKF45' solver class
    """

    def test_init(self):

        #test default initializtion
        solver = RKF45()

        self.assertEqual(solver.initial_value, 0)

        self.assertEqual(solver.stage, 0)
        self.assertTrue(solver.is_adaptive)
        self.assertTrue(solver.is_explicit)
        self.assertFalse(solver.is_implicit)
        
        #test specific initialization
        solver = RKF45(
            initial_value=1, 
            tolerance_lte_rel=1e-3, 
            tolerance_lte_abs=1e-6
            )

        self.assertEqual(solver.initial_value, 1)
        self.assertEqual(solver.tolerance_lte_rel, 1e-3)
        self.assertEqual(solver.tolerance_lte_abs, 1e-6)


    def test_stages(self):

        solver = RKF45()

        for i, t in enumerate(solver.stages(0, 1)):
            
            #test the stage iterator
            self.assertEqual(t, solver.eval_stages[i])


    def test_step(self):

        solver = RKF45()
        
        solver.buffer(1)

        for i, t in enumerate(solver.stages(0, 1)):

            #test if stage incrementation works
            self.assertEqual(solver.stage, i)

            success, err, scale = solver.step(0.0, 1)

            #test if expected return at intermediate stages
            if i < len(solver.eval_stages)-1:
                self.assertTrue(success)
                self.assertEqual(err, 0.0)
                self.assertIsNone(scale)  # No rescale needed at intermediate stages

        #test if expected return at final stage
        self.assertNotEqual(err, 0.0)
        self.assertIsNotNone(scale)  # Actual scale at final stage


    def test_integrate_fixed(self):
        
        #dict for logging
        stats = {}
        
        #divisons of integration duration
        divisions = np.logspace(1, 3, 20)

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = RKF45(problem.x0)
                
                errors = []

                timesteps = (problem.t_span[1] - problem.t_span[0]) / divisions

                for dt in timesteps:

                    solver.reset()
                    time, numerical_solution = solver.integrate(
                        problem.func, 
                        time_start=problem.t_span[0], 
                        time_end=problem.t_span[1], 
                        dt=dt, 
                        adaptive=False
                        )

                    analytical_solution = problem.solution(time)
                    err = np.mean(abs(numerical_solution - analytical_solution))
                    errors.append(err)

                errors = np.array(errors)
                timesteps = np.array(timesteps)

                #restrict the convergence assessment to the region above the
                #roundoff floor: now that RKF45 propagates the 5th order weights
                #it drives some reference problems to machine precision at fine
                #dt, where the error saturates and is no longer a clean power law
                above_floor = errors > 1e-11
                errs_fit = errors[above_floor]
                dts_fit = timesteps[above_floor]

                #test if errors are monotonically decreasing (above the floor)
                self.assertTrue(np.all(np.diff(errs_fit) < 0))

                #test convergence order (global). A conservative n-2 bound is
                #used here because the now-5th-order propagation makes a plain
                #fixed-step fit sensitive to fixture effects on the stiffer
                #reference problems (rapidly growing higher derivatives near the
                #blow-up of 1/(1-t) pollute the asymptotic slope). The clean
                #order fit lives in test_convergence_order_fixed_step.
                p, _ = np.polyfit(np.log10(dts_fit), np.log10(errs_fit), deg=1)
                self.assertGreater(p, solver.n-2)

            #log stats
            stats[problem.name] = {"n":p, "err":errors, "dt":timesteps}

        # fig, ax = plt.subplots(dpi=120, tight_layout=True)
        # fig.suptitle(solver.__class__.__name__)
        # for name, stat in stats.items(): 
        #     ax.loglog(stat["dt"], stat["err"], label=name)
        # ax.loglog(timesteps, timesteps**solver.n, c="k", ls="--", label=f"n={solver.n}")
        # ax.legend()
        # plt.show()


    def test_convergence_order_fixed_step(self):

        #RKF45 propagates the 5th order weights -> fixed-step global convergence
        #order must be ~5 on a smooth problem. Driven step-by-step to an exact
        #endpoint (no overshoot) so the fitted slope reflects the tableau order.
        #x' = x, x(0) = 1  ->  exact x(1) = e.
        func = lambda x, t: x
        exact = np.exp(1.0)

        n_steps = [10, 20, 40, 80, 160]
        dts = [1.0 / n for n in n_steps]
        errors = []
        for n in n_steps:
            solver = RKF45(1.0)
            solver.reset()
            t, dt = 0.0, 1.0 / n
            for _ in range(n):
                solver.integrate_singlestep(func, time=t, dt=dt)
                t += dt
            errors.append(abs(float(np.ravel(solver.x)[0]) - exact))

        #errors must decrease monotonically
        self.assertTrue(np.all(np.diff(errors) < 0))

        #least-squares order fit must be close to 5 (the pre-fix 4th order
        #propagation gives ~4; a broken high-order row would give < 4)
        p, _ = np.polyfit(np.log10(dts), np.log10(errors), deg=1)
        self.assertGreater(p, 4.6)
        self.assertLess(p, 5.4)


    def test_integrate_adaptive(self):

        #integrate test problem and assess convergence order
        for problem in PROBLEMS:

            with self.subTest(problem.name):

                solver = RKF45(problem.x0, tolerance_lte_rel=0, tolerance_lte_abs=1e-5)

                duration = problem.t_span[1] - problem.t_span[0]
                
                time, numerical_solution = solver.integrate(
                    problem.func, 
                    time_start=problem.t_span[0], 
                    time_end=problem.t_span[1], 
                    dt=duration/100, 
                    adaptive=True
                    )

                analytical_solution = problem.solution(time)
                err = np.mean(abs(numerical_solution - analytical_solution))

                #test if error control was successful (same OOM for global error -> < 1e-5)
                self.assertLess(err, solver.tolerance_lte_abs*10)



# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':

    unittest.main(verbosity=2)