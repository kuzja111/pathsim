########################################################################################
##
##                EXPLICIT ADAPTIVE TIMESTEPPING RUNGE-KUTTA INTEGRATORS
##                                 (solvers/rkf45.py)
##
##                                 Milan Rother 2024
##
########################################################################################

# IMPORTS ==============================================================================

from ._rungekutta import ExplicitRungeKutta


# SOLVERS ==============================================================================

class RKF45(ExplicitRungeKutta):
    """Runge-Kutta-Fehlberg 5(4) pair. Six stages, 5th order propagation with
    an embedded 4th order solution for local error estimation (local
    extrapolation, matching the ``RKCK54`` / ``RKDP54`` convention).

    The historically first widely-used embedded pair for automatic step-size
    control. The 5th order solution ``b5`` is propagated; the difference to the
    4th order solution ``b4`` (``TR = b5 - b4``) provides a local error estimate
    of the lower-order result.

    Characteristics
    ---------------
    * Order: 5 (propagating) / 4 (error estimate)
    * Stages: 6
    * Explicit, adaptive timestep

    Note
    ----
    Largely superseded by the Dormand-Prince (``RKDP54``) and Cash-Karp
    (``RKCK54``) pairs, which achieve better accuracy per function evaluation
    on most problems. Still useful for reproducing legacy results or when
    comparing against published benchmarks that used RKF45.

    References
    ----------
    .. [1] Fehlberg, E. (1969). "Low-order classical Runge-Kutta formulas
           with stepsize control and their application to some heat transfer
           problems". NASA Technical Report TR R-315.
    .. [2] Fehlberg, E. (1970). "Klassische Runge-Kutta-Formeln vierter und
           niedrigerer Ordnung mit Schrittweiten-Kontrolle und ihre Anwendung
           auf Wärmeleitungsprobleme". Computing, 6(1-2), 61-71.
           :doi:`10.1007/BF02241732`

    """

    def __init__(self, *solver_args, **solver_kwargs):
        super().__init__(*solver_args, **solver_kwargs)

        #number of stages in RK scheme
        self.s = 6

        #order of scheme and embedded method
        self.n = 5
        self.m = 4

        #flag adaptive timestep solver
        self.is_adaptive = True

        #intermediate evaluation times
        self.eval_stages = [0.0, 1/4, 3/8, 12/13, 1, 1/2]

        #extended butcher table (propagates the 5th order weights b5)
        self.BT = {
            0: [      1/4],
            1: [     3/32,       9/32],
            2: [1932/2197, -7200/2197,   7296/2197],
            3: [  439/216,         -8,    3680/513,  -845/4104],
            4: [    -8/27,          2,  -3544/2565,  1859/4104,  -11/40],
            5: [   16/135,          0, 6656/12825, 28561/56430,   -9/50, 2/55]
            }

        #coefficients for local truncation error estimate (TR = b5 - b4)
        self.TR = [1/360, 0, -128/4275, -2197/75240, 1/50, 2/55]