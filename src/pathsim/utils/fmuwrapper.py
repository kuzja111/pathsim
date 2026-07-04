#########################################################################################
##
##                   FMU WRAPPER - VERSION AGNOSTIC FMI INTERFACE
##                            (pathsim/utils/fmuwrapper.py)
##
#########################################################################################

# IMPORTS ===============================================================================

import numpy as np
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

from .register import Register


# HELPER CLASSES ========================================================================

@dataclass
class EventInfo:
    """Unified event information structure for both FMI 2.0 and 3.0.

    Attributes
    ----------
    discrete_states_need_update : bool
        whether discrete state iteration is needed
    terminate_simulation : bool
        whether FMU requests simulation termination
    nominals_changed : bool
        whether nominal values of continuous states changed
    values_changed : bool
        whether continuous state values changed
    next_event_time_defined : bool
        whether FMU has scheduled a next time event
    next_event_time : float
        time of next scheduled event (if defined)
    """
    discrete_states_need_update: bool = False
    terminate_simulation: bool = False
    nominals_changed: bool = False
    values_changed: bool = False
    next_event_time_defined: bool = False
    next_event_time: float = 0.0


@dataclass
class StepResult:
    """Result information from a co-simulation step.

    Attributes
    ----------
    event_encountered : bool
        whether an event was encountered during step (FMI 3.0 only)
    terminate_simulation : bool
        whether FMU requests simulation termination (FMI 3.0 only)
    early_return : bool
        whether step returned early (FMI 3.0 only)
    last_successful_time : float
        last time successfully reached (FMI 3.0 only)
    """
    event_encountered: bool = False
    terminate_simulation: bool = False
    early_return: bool = False
    last_successful_time: float = 0.0


# FMI VERSION-SPECIFIC OPERATIONS =======================================================

class _FMI2Ops:
    """FMI 2.0 specific operations."""

    @staticmethod
    def set_real(fmu, refs, values):
        fmu.setReal(refs, values)

    @staticmethod
    def get_real(fmu, refs):
        return fmu.getReal(refs)

    @staticmethod
    def set_integer(fmu, refs, values):
        fmu.setInteger(refs, values)

    @staticmethod
    def get_integer(fmu, refs):
        return fmu.getInteger(refs)

    @staticmethod
    def do_step(fmu, current_time, step_size):
        fmu.doStep(current_time, step_size)
        return StepResult()

    @staticmethod
    def get_derivatives(fmu, n_states):
        if n_states == 0:
            return np.array([])
        derivatives = (ctypes.c_double * n_states)()
        fmu.getDerivatives(derivatives, n_states)
        return np.array(derivatives)

    @staticmethod
    def update_discrete_states(fmu):
        result = fmu.newDiscreteStates()
        return EventInfo(
            discrete_states_need_update=result[0],
            terminate_simulation=result[1],
            nominals_changed=result[2],
            values_changed=result[3],
            next_event_time_defined=result[4],
            next_event_time=result[5]
        )

    @staticmethod
    def setup_experiment(fmu, tolerance, start_time, stop_time):
        fmu.setupExperiment(tolerance=tolerance, startTime=start_time, stopTime=stop_time)

    @staticmethod
    def enter_initialization_mode(fmu, tolerance, start_time, stop_time):
        fmu.enterInitializationMode()

    @staticmethod
    def exit_initialization_mode(fmu, mode):
        result = fmu.exitInitializationMode()
        # FMI 2.0 doesn't return event info from exitInitializationMode
        return None


class _FMI3Ops:
    """FMI 3.0 specific operations."""

    @staticmethod
    def set_real(fmu, refs, values):
        fmu.setFloat64(refs, values)

    @staticmethod
    def get_real(fmu, refs):
        return fmu.getFloat64(refs)

    @staticmethod
    def set_integer(fmu, refs, values):
        fmu.setInt64(refs, values)

    @staticmethod
    def get_integer(fmu, refs):
        return fmu.getInt64(refs)

    @staticmethod
    def do_step(fmu, current_time, step_size):
        event, terminate, early, last_time = fmu.doStep(current_time, step_size)
        return StepResult(
            event_encountered=event,
            terminate_simulation=terminate,
            early_return=early,
            last_successful_time=last_time
        )

    @staticmethod
    def get_derivatives(fmu, n_states):
        if n_states == 0:
            return np.array([])
        derivatives = (ctypes.c_double * n_states)()
        fmu.getContinuousStateDerivatives(derivatives, n_states)
        return np.array(derivatives)

    @staticmethod
    def update_discrete_states(fmu):
        result = fmu.updateDiscreteStates()
        return EventInfo(
            discrete_states_need_update=result[0],
            terminate_simulation=result[1],
            nominals_changed=result[2],
            values_changed=result[3],
            next_event_time_defined=result[4],
            next_event_time=result[5]
        )

    @staticmethod
    def setup_experiment(fmu, tolerance, start_time, stop_time):
        # FMI 3.0 passes these to enterInitializationMode instead
        pass

    @staticmethod
    def enter_initialization_mode(fmu, tolerance, start_time, stop_time):
        fmu.enterInitializationMode(tolerance=tolerance, startTime=start_time, stopTime=stop_time)

    @staticmethod
    def exit_initialization_mode(fmu, mode):
        result = fmu.exitInitializationMode()
        # FMI 3.0 Model Exchange returns event info
        if mode == 'model_exchange' and hasattr(result, 'nextEventTimeDefined'):
            return EventInfo(
                discrete_states_need_update=bool(getattr(result, 'discreteStatesNeedUpdate', False)),
                terminate_simulation=bool(getattr(result, 'terminateSimulation', False)),
                nominals_changed=bool(getattr(result, 'nominalsOfContinuousStatesChanged', False)),
                values_changed=bool(getattr(result, 'valuesOfContinuousStatesChanged', False)),
                next_event_time_defined=bool(getattr(result, 'nextEventTimeDefined', False)),
                next_event_time=float(getattr(result, 'nextEventTime', 0.0))
            )
        return None


# MAIN WRAPPER CLASS ====================================================================

class FMUWrapper:
    """Version-agnostic wrapper for FMI 2.0 and 3.0 FMUs.

    This class provides a unified interface for working with FMUs regardless of
    FMI version (2.0 or 3.0) or interface type (Co-Simulation or Model Exchange).
    It handles all version-specific API differences internally.

    Parameters
    ----------
    fmu_path : str
        path to the FMU file (.fmu)
    instance_name : str, optional
        name for the FMU instance (default: 'fmu_instance')
    mode : str, optional
        FMU interface mode: 'cosimulation' or 'model_exchange' (default: 'cosimulation')

    Attributes
    ----------
    fmu_path : str
        path to the FMU file
    instance_name : str
        name of the FMU instance
    mode : str
        interface mode ('cosimulation' or 'model_exchange')
    model_description : ModelDescription
        FMI model description from FMPy (use this for metadata access)
    fmu : FMU2Slave | FMU3Slave | FMU2Model | FMU3Model
        underlying FMPy FMU instance
    fmi_version : str
        detected FMI version ('2.0' or '3.0')
    n_states : int
        number of continuous states (Model Exchange only)
    n_event_indicators : int
        number of event indicators (Model Exchange only)
    input_refs : dict
        mapping from input variable names to value references
    output_refs : dict
        mapping from output variable names to value references
    """

    def __init__(self, fmu_path, instance_name="fmu_instance", mode="cosimulation"):

        # Import FMPy (lazy import to avoid dependency if not used)
        try:
            from fmpy import read_model_description, extract
            from fmpy.fmi2 import FMU2Slave, FMU2Model
            from fmpy.fmi3 import FMU3Slave, FMU3Model
        except ImportError:
            raise ImportError("FMPy is required for FMU support. Install with: pip install fmpy")

        self.fmu_path = fmu_path
        self.instance_name = instance_name
        self.mode = mode.lower()

        if self.mode not in ['cosimulation', 'model_exchange']:
            raise ValueError(f"Invalid mode '{mode}'. Must be 'cosimulation' or 'model_exchange'")

        # Read model description and detect FMI version
        self.model_description = read_model_description(fmu_path)
        self.fmi_version = self.model_description.fmiVersion

        # Select version-specific operations
        self._ops = _FMI2Ops if self.fmi_version.startswith('2.') else _FMI3Ops

        # Extract FMU
        self.unzipdir = extract(fmu_path)

        # Build variable lookup maps
        self._build_variable_maps()

        # Get state and event info for Model Exchange
        if self.mode == 'model_exchange':
            self.n_states = self.model_description.numberOfContinuousStates
            self.n_event_indicators = self.model_description.numberOfEventIndicators
            self._build_state_derivative_maps()
        else:
            self.n_states = 0
            self.n_event_indicators = 0
            self._state_refs = []
            self._derivative_refs = []

        # Instantiate appropriate FMU class based on version and mode
        self.fmu = self._create_fmu_instance(FMU2Slave, FMU2Model, FMU3Slave, FMU3Model)

    def _create_fmu_instance(self, FMU2Slave, FMU2Model, FMU3Slave, FMU3Model):
        """Create the appropriate FMU instance based on version and mode."""
        md = self.model_description

        if self.fmi_version.startswith('2.'):
            if self.mode == 'cosimulation':
                return FMU2Slave(
                    guid=md.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=md.coSimulation.modelIdentifier,
                    instanceName=self.instance_name
                )
            else:
                return FMU2Model(
                    guid=md.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=md.modelExchange.modelIdentifier,
                    instanceName=self.instance_name
                )
        elif self.fmi_version.startswith('3.'):
            if self.mode == 'cosimulation':
                return FMU3Slave(
                    guid=md.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=md.coSimulation.modelIdentifier,
                    instanceName=self.instance_name
                )
            else:
                return FMU3Model(
                    guid=md.guid,
                    unzipDirectory=self.unzipdir,
                    modelIdentifier=md.modelExchange.modelIdentifier,
                    instanceName=self.instance_name
                )
        else:
            raise ValueError(f"Unsupported FMI version: {self.fmi_version}")

    def _build_variable_maps(self):
        """Build internal variable name to reference mappings."""
        self.variable_map = {var.name: var for var in self.model_description.modelVariables}
        self.input_refs = {}
        self.output_refs = {}

        for variable in self.model_description.modelVariables:
            if variable.causality == 'input':
                self.input_refs[variable.name] = variable.valueReference
            elif variable.causality == 'output':
                self.output_refs[variable.name] = variable.valueReference

    def _build_state_derivative_maps(self):
        """Build state and derivative value reference lists (Model Exchange only).

        In FMI, state variables have a 'derivative' attribute pointing to their
        derivative variable. This method extracts the ordered lists of value
        references needed for Jacobian computation via directional derivatives.
        """
        self._state_refs = []
        self._derivative_refs = []

        for var in self.model_description.modelVariables:
            if var.derivative is not None:
                # This variable is a state (it has a derivative)
                self._state_refs.append(var.valueReference)
                self._derivative_refs.append(var.derivative.valueReference)

    # ===================================================================================
    # CONVENIENCE METHODS FOR BLOCK INITIALIZATION
    # ===================================================================================

    def create_port_registers(self) -> Tuple[Register, Register]:
        """Create input and output registers for block I/O.

        Returns
        -------
        inputs : Register
            input register with FMU input variable names as labels
        outputs : Register
            output register with FMU output variable names as labels
        """
        port_map_in = {name: idx for idx, name in enumerate(self.input_refs.keys())}
        port_map_out = {name: idx for idx, name in enumerate(self.output_refs.keys())}

        inputs = Register(size=len(port_map_in), mapping=port_map_in)
        outputs = Register(size=len(port_map_out), mapping=port_map_out)

        return inputs, outputs

    def initialize(self, start_values=None, start_time=0.0, stop_time=None,
                   tolerance=None) -> Optional[EventInfo]:
        """Complete FMU initialization sequence.

        Performs: instantiate -> setup_experiment -> enter_initialization_mode
        -> set start values -> exit_initialization_mode

        Parameters
        ----------
        start_values : dict, optional
            dictionary of variable names and their initial values
        start_time : float, optional
            simulation start time (default: 0.0)
        stop_time : float, optional
            simulation stop time
        tolerance : float, optional
            tolerance for integration/event detection

        Returns
        -------
        event_info : EventInfo or None
            event information for FMI 3.0 Model Exchange, None otherwise
        """
        self.instantiate()
        self.setup_experiment(tolerance=tolerance, start_time=start_time, stop_time=stop_time)
        self.enter_initialization_mode()

        if start_values:
            for name, value in start_values.items():
                self.set_variable(name, value)

        return self.exit_initialization_mode()

    @property
    def default_step_size(self) -> Optional[float]:
        """Get default step size from FMU's default experiment, if defined."""
        de = self.model_description.defaultExperiment
        if de is not None:
            step_size = getattr(de, 'stepSize', None)
            return float(step_size) if step_size is not None else None
        return None

    @property
    def default_tolerance(self) -> Optional[float]:
        """Get default tolerance from FMU's default experiment, if defined."""
        de = self.model_description.defaultExperiment
        if de is not None:
            tolerance = getattr(de, 'tolerance', None)
            return float(tolerance) if tolerance is not None else None
        return None

    @property
    def needs_completed_integrator_step(self) -> bool:
        """Check if FMU requires completedIntegratorStep notifications (Model Exchange only)."""
        if self.mode != 'model_exchange':
            return False
        me = self.model_description.modelExchange
        return not getattr(me, 'completedIntegratorStepNotNeeded', False)

    @property
    def provides_jacobian(self) -> bool:
        """Check if FMU provides directional derivatives for Jacobian computation."""
        if self.mode == 'model_exchange':
            me = self.model_description.modelExchange
            return getattr(me, 'providesDirectionalDerivative', False)
        elif self.mode == 'cosimulation':
            cs = self.model_description.coSimulation
            return getattr(cs, 'providesDirectionalDerivative', False)
        return False

    def get_state_jacobian(self):
        """Compute Jacobian of state derivatives w.r.t. states (Model Exchange only).

        Uses FMU's directional derivative capability to compute ∂ẋ/∂x.
        Requires the FMU to have providesDirectionalDerivative=true.

        Returns
        -------
        jacobian : np.ndarray
            n_states x n_states Jacobian matrix, or None if not supported
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("get_state_jacobian() is only available for Model Exchange FMUs")

        if not self.provides_jacobian:
            return None

        if self.n_states == 0:
            return np.array([]).reshape(0, 0)

        # Build Jacobian column by column using directional derivatives
        jacobian = np.zeros((self.n_states, self.n_states))
        seed = np.zeros(self.n_states)

        for j in range(self.n_states):
            seed[j] = 1.0
            col = self.fmu.getDirectionalDerivative(
                self._derivative_refs,
                self._state_refs,
                seed.tolist()
            )
            jacobian[:, j] = col
            seed[j] = 0.0

        return jacobian

    # ===================================================================================
    # FMU LIFECYCLE METHODS
    # ===================================================================================

    def instantiate(self, visible=False, logging_on=False):
        """Instantiate the FMU."""
        self.fmu.instantiate(visible=visible, loggingOn=logging_on)

    def setup_experiment(self, tolerance=None, start_time=0.0, stop_time=None):
        """Setup experiment parameters."""
        self._tolerance = tolerance
        self._start_time = start_time
        self._stop_time = stop_time
        self._ops.setup_experiment(self.fmu, tolerance, start_time, stop_time)

    def enter_initialization_mode(self):
        """Enter initialization mode."""
        self._ops.enter_initialization_mode(
            self.fmu, self._tolerance, self._start_time, self._stop_time
        )

    def exit_initialization_mode(self) -> Optional[EventInfo]:
        """Exit initialization mode and return event information."""
        return self._ops.exit_initialization_mode(self.fmu, self.mode)

    def reset(self):
        """Reset FMU to initial state."""
        self.fmu.reset()

    def terminate(self):
        """Terminate FMU."""
        self.fmu.terminate()

    def free_instance(self):
        """Free FMU instance and resources."""
        self.fmu.freeInstance()

    # ===================================================================================
    # VARIABLE ACCESS METHODS
    # ===================================================================================

    def set_real(self, refs, values):
        """Set real-valued variables by reference."""
        values = np.atleast_1d(values)
        self._ops.set_real(self.fmu, refs, values)

    def get_real(self, refs):
        """Get real-valued variables by reference."""
        return np.array(self._ops.get_real(self.fmu, refs))

    def set_variable(self, name, value):
        """Set a single variable by name (automatically detects type)."""
        variable = self.variable_map.get(name)
        if variable is None:
            raise ValueError(f"Variable '{name}' not found in FMU")

        vr = variable.valueReference
        var_type = variable.type

        if var_type in ['Real', 'Float64', 'Float32']:
            self._ops.set_real(self.fmu, [vr], [float(value)])
        elif var_type in ['Integer', 'Int64', 'Int32', 'Int16', 'Int8']:
            self._ops.set_integer(self.fmu, [vr], [int(value)])
        elif var_type == 'Boolean':
            self.fmu.setBoolean([vr], [bool(value)])
        else:
            raise ValueError(f"Unsupported variable type: {var_type}")

    def set_inputs_from_array(self, values):
        """Set all FMU inputs from an array."""
        if len(self.input_refs) > 0:
            input_vrefs = list(self.input_refs.values())
            self.set_real(input_vrefs, values)

    def get_outputs_as_array(self):
        """Get all FMU outputs as an array."""
        if len(self.output_refs) == 0:
            return np.array([])
        output_vrefs = list(self.output_refs.values())
        return self.get_real(output_vrefs)

    # ===================================================================================
    # CO-SIMULATION METHODS
    # ===================================================================================

    def do_step(self, current_time, step_size) -> StepResult:
        """Perform a co-simulation step."""
        if self.mode != 'cosimulation':
            raise RuntimeError("do_step() is only available for Co-Simulation FMUs")
        return self._ops.do_step(self.fmu, current_time, step_size)

    # ===================================================================================
    # MODEL EXCHANGE METHODS
    # ===================================================================================

    def set_time(self, time):
        """Set current time (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("set_time() is only available for Model Exchange FMUs")
        self.fmu.setTime(time)

    def set_continuous_states(self, states):
        """Set continuous states (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("set_continuous_states() is only available for Model Exchange FMUs")
        if self.n_states == 0:
            return
        states = np.atleast_1d(states)
        x_ctypes = (ctypes.c_double * self.n_states)(*states)
        self.fmu.setContinuousStates(x_ctypes, self.n_states)

    def get_continuous_states(self):
        """Get continuous states (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("get_continuous_states() is only available for Model Exchange FMUs")
        if self.n_states == 0:
            return np.array([])
        states = (ctypes.c_double * self.n_states)()
        self.fmu.getContinuousStates(states, self.n_states)
        return np.array(states)

    def get_derivatives(self):
        """Get state derivatives (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("get_derivatives() is only available for Model Exchange FMUs")
        return self._ops.get_derivatives(self.fmu, self.n_states)

    def get_event_indicators(self):
        """Get event indicators (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("get_event_indicators() is only available for Model Exchange FMUs")
        if self.n_event_indicators == 0:
            return np.array([])
        indicators = (ctypes.c_double * self.n_event_indicators)()
        self.fmu.getEventIndicators(indicators, self.n_event_indicators)
        return np.array(indicators)

    def enter_event_mode(self):
        """Enter event mode (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("enter_event_mode() is only available for Model Exchange FMUs")
        self.fmu.enterEventMode()

    def enter_continuous_time_mode(self):
        """Enter continuous time mode (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("enter_continuous_time_mode() is only available for Model Exchange FMUs")
        self.fmu.enterContinuousTimeMode()

    def update_discrete_states(self) -> EventInfo:
        """Update discrete states during event iteration (Model Exchange only)."""
        if self.mode != 'model_exchange':
            raise RuntimeError("update_discrete_states() is only available for Model Exchange FMUs")
        return self._ops.update_discrete_states(self.fmu)

    def completed_integrator_step(self) -> Tuple[bool, bool]:
        """Notify FMU that integrator step completed (Model Exchange only).

        Returns
        -------
        enter_event_mode : bool
            whether FMU requests event mode
        terminate_simulation : bool
            whether FMU requests simulation termination
        """
        if self.mode != 'model_exchange':
            raise RuntimeError("completed_integrator_step() is only available for Model Exchange FMUs")
        return self.fmu.completedIntegratorStep()

    def __del__(self):
        """Cleanup FMU resources on deletion."""
        try:
            self.terminate()
            self.free_instance()
        except:
            pass
