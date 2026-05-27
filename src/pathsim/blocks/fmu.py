#########################################################################################
##
##                           FUNCTIONAL MOCK-UP UNIT (FMU) BLOCKS
##                                   (pathsim/blocks/fmu.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import bisect

from ._block import Block
from .dynsys import DynamicalSystem

from ..events.schedule import Schedule, ScheduleList
from ..events.zerocrossing import ZeroCrossing
from ..utils.fmuwrapper import FMUWrapper
from ..utils.deprecation import deprecated


# BLOCKS ================================================================================

@deprecated(
    version="1.0.0",
    replacement="pathsim_fmi.CoSimulationFMU",
    reason="This block has moved to the pathsim-fmi package: pip install pathsim-fmi",
)
class CoSimulationFMU(Block):
    """Co-Simulation FMU block using FMPy with support for FMI 2.0 and FMI 3.0.

    This block wraps an FMU (Functional Mock-up Unit) for co-simulation.
    The FMU encapsulates a simulation model that can be executed independently
    and synchronized with the main simulation at discrete communication points.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    start_values : dict, optional
        dictionary of variable names and their initial values
    dt : float, optional
        communication step size for co-simulation. If None, uses the FMU's
        default experiment step size if available.

    Attributes
    ----------
    fmu_wrapper : FMUWrapper
        version-agnostic FMU wrapper instance providing access to model_description,
        fmu, and other FMPy objects for advanced usage
    dt : float
        communication step size
    """

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None, dt=None):
        super().__init__()

        self.start_values = start_values

        # Create and initialize FMU wrapper
        self.fmu_wrapper = FMUWrapper(fmu_path, instance_name, mode='cosimulation')
        self.fmu_wrapper.initialize(start_values, start_time=0.0)

        # Determine step size
        self.dt = dt if dt is not None else self.fmu_wrapper.default_step_size
        if self.dt is None:
            raise ValueError("No step size provided and FMU has no default experiment step size")

        # Setup block I/O from FMU variables
        self.inputs, self.outputs = self.fmu_wrapper.create_port_registers()

        # Scheduled co-simulation step
        self.events = [
            Schedule(
                t_start=0, 
                t_period=self.dt, 
                func_act=self._step_fmu
                )
            ]

        # Read initial outputs
        self.outputs.update_from_array(self.fmu_wrapper.get_outputs_as_array())


    def _step_fmu(self, t):
        """Perform one FMU co-simulation step."""
        self.fmu_wrapper.set_inputs_from_array(self.inputs.to_array())

        result = self.fmu_wrapper.do_step(
            current_time=t, 
            step_size=self.dt
            )

        if result.terminate_simulation:
            raise RuntimeError("FMU requested simulation termination")

        self.outputs.update_from_array(self.fmu_wrapper.get_outputs_as_array())


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu_wrapper.reset()
        self.fmu_wrapper.initialize(self.start_values, start_time=0.0)
        self.outputs.update_from_array(self.fmu_wrapper.get_outputs_as_array())


    def __len__(self):
        """FMU is a discrete time source-like block without direct passthrough."""
        return 0


@deprecated(
    version="1.0.0",
    replacement="pathsim_fmi.ModelExchangeFMU",
    reason="This block has moved to the pathsim-fmi package: pip install pathsim-fmi",
)
class ModelExchangeFMU(DynamicalSystem):
    """Model Exchange FMU block using FMPy with support for FMI 2.0 and FMI 3.0.

    This block wraps an FMU (Functional Mock-up Unit) for model exchange.
    The FMU provides the right-hand side of an ODE system that is integrated
    by PathSim's numerical solvers. Internal FMU events (state events, time
    events, and step completion events) are translated to PathSim events.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    start_values : dict, optional
        dictionary of variable names and their initial values
    tolerance : float, optional
        tolerance for event detection (default: 1e-10)
    verbose : bool, optional
        enable verbose output (default: False)

    Attributes
    ----------
    fmu_wrapper : FMUWrapper
        version-agnostic FMU wrapper instance providing access to model_description,
        fmu, and other FMPy objects for advanced usage
    time_event : ScheduleList or None
        dynamic time event for FMU-scheduled events
    """

    def __init__(self, fmu_path, instance_name="fmu_instance", start_values=None,
                 tolerance=1e-10, verbose=False):

        self.tolerance = tolerance
        self.verbose = verbose
        self.start_values = start_values

        # Create and initialize FMU wrapper
        self.fmu_wrapper = FMUWrapper(fmu_path, instance_name, mode='model_exchange')
        event_info = self.fmu_wrapper.initialize(start_values, start_time=0.0, tolerance=tolerance)

        # Store initial time event if defined
        self._initial_time_event = (
            event_info.next_event_time
            if event_info and event_info.next_event_time_defined
            else None
        )

        # Enter continuous time mode
        self.fmu_wrapper.enter_continuous_time_mode()

        # Initialize parent DynamicalSystem with FMU dynamics
        # Use FMU's Jacobian if available (providesDirectionalDerivative=true)
        jac_func = self._get_jacobian if self.fmu_wrapper.provides_jacobian else None

        super().__init__(
            func_dyn=self._get_derivatives,
            func_alg=self._get_outputs,
            initial_value=self.fmu_wrapper.get_continuous_states(),
            jac_dyn=jac_func
        )

        # Setup block I/O from FMU variables
        self.inputs, self.outputs = self.fmu_wrapper.create_port_registers()

        # Initialize time event manager
        self.time_event = None

        # Create state event (zero-crossing) for each event indicator
        for i in range(self.fmu_wrapper.n_event_indicators):
            self.events.append(
                ZeroCrossing(
                    func_evt=lambda t, idx=i: self._get_event_indicator(idx),
                    func_act=self._handle_event,
                    tolerance=self.tolerance
                    )
                )

        # Cache capability flag for sample() performance
        self._needs_completed_integrator_step = self.fmu_wrapper.needs_completed_integrator_step

        # Schedule initial time event if any
        if self._initial_time_event is not None:
            self._update_time_events(self._initial_time_event)


    def _get_derivatives(self, x, u, t):
        """Evaluate FMU derivatives (RHS of ODE)."""
        if self.fmu_wrapper.n_states == 0:
            return []

        self.fmu_wrapper.set_time(t)
        self.fmu_wrapper.set_continuous_states(x)
        self.fmu_wrapper.set_inputs_from_array(u)

        return self.fmu_wrapper.get_derivatives()


    def _get_jacobian(self, x, u, t):
        """Evaluate Jacobian of FMU derivatives w.r.t. states (∂ẋ/∂x)."""
        self.fmu_wrapper.set_time(t)
        self.fmu_wrapper.set_continuous_states(x)
        self.fmu_wrapper.set_inputs_from_array(u)

        return self.fmu_wrapper.get_state_jacobian()


    def _get_outputs(self, x, u, t):
        """Evaluate FMU outputs (algebraic part)."""
        self.fmu_wrapper.set_time(t)
        self.fmu_wrapper.set_continuous_states(x)
        self.fmu_wrapper.set_inputs_from_array(u)

        return self.fmu_wrapper.get_outputs_as_array()


    def _get_event_indicator(self, idx):
        """Get value of a specific event indicator."""
        return self.fmu_wrapper.get_event_indicators()[idx]


    def _handle_event(self, t):
        """Handle FMU event with fixed-point iteration for discrete states."""
        if self.verbose:
            print(f"FMU event detected at t={t}")

        self.fmu_wrapper.enter_event_mode()

        # Iterate until discrete states stabilize
        while True:
            event_info = self.fmu_wrapper.update_discrete_states()

            if event_info.terminate_simulation:
                raise RuntimeError("FMU requested simulation termination")

            if not event_info.discrete_states_need_update:
                break

        self.fmu_wrapper.enter_continuous_time_mode()

        # Update continuous states if changed
        if event_info.values_changed:
            x_new = self.fmu_wrapper.get_continuous_states()
            self.engine.set(x_new)
            if self.verbose:
                print(f"Continuous states updated after event: {x_new}")

        # Schedule new time events
        if event_info.next_event_time_defined:
            self._update_time_events(event_info.next_event_time)
            if self.verbose:
                print(f"Next time event scheduled at t={event_info.next_event_time}")


    def _update_time_events(self, next_time):
        """Update or create time event schedule."""
        if self.time_event is None:
            self.time_event = ScheduleList(
                times_evt=[next_time],
                func_act=self._handle_event,
                tolerance=self.tolerance
            )
            self.events.append(self.time_event)
        elif next_time not in self.time_event.times_evt:
            bisect.insort(self.time_event.times_evt, next_time)


    def sample(self, t, dt):
        """Sample block after successful timestep and handle FMU step completion events."""
        super().sample(t, dt)

        if self._needs_completed_integrator_step:
            enter_event_mode, terminate_simulation = self.fmu_wrapper.completed_integrator_step()

            if terminate_simulation:
                raise RuntimeError("FMU requested simulation termination")

            if enter_event_mode:
                if self.verbose:
                    print(f"Step completion event at t={t}")
                self._handle_event(t)


    def reset(self):
        """Reset the FMU instance."""
        super().reset()
        self.fmu_wrapper.reset()

        # Re-initialize FMU
        event_info = self.fmu_wrapper.initialize(
            self.start_values, start_time=0.0, tolerance=self.tolerance
        )
        self.fmu_wrapper.enter_continuous_time_mode()

        # Reset to initial states
        self.engine.set(self.fmu_wrapper.get_continuous_states())

        # Reset time events
        if self.time_event is not None:
            self.time_event.times_evt.clear()

        # Schedule initial time event from re-initialization or cached initial
        if event_info and event_info.next_event_time_defined:
            self._update_time_events(event_info.next_event_time)
        elif self._initial_time_event is not None:
            self._update_time_events(self._initial_time_event)