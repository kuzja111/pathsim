########################################################################################
##
##                                  TESTS FOR
##                            'utils.fmuwrapper.py'
##
########################################################################################

# IMPORTS ==============================================================================

import unittest
import numpy as np
import os
import platform

from pathlib import Path


# Path to test FMU files
TEST_DATA_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "source" / "examples" / "data"

# Co-simulation test FMU per host platform
_PLATFORM_FMU_NAMES = {
    "Windows": "CoupledClutches_CS_win64.fmu",
    "Linux": "CoupledClutches_CS_linux64.fmu",
}


# Helper to check if FMPy is available
def fmpy_available():
    try:
        import fmpy
        return True
    except ImportError:
        return False


# TESTS ================================================================================

class TestFMUWrapperImport(unittest.TestCase):
    """Test FMUWrapper import behavior"""

    def test_import_error_without_fmpy(self):
        """Test that ImportError is raised when FMPy is not available"""
        if fmpy_available():
            self.skipTest("FMPy is installed, cannot test ImportError case")

        from pathsim.utils.fmuwrapper import FMUWrapper

        with self.assertRaises(ImportError) as context:
            wrapper = FMUWrapper("nonexistent.fmu")

        self.assertIn("FMPy", str(context.exception))


@unittest.skipIf(not fmpy_available(), "FMPy not installed")
@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestFMUWrapperCoSimulation(unittest.TestCase):
    """Test FMUWrapper with Co-Simulation FMU"""

    @classmethod
    def setUpClass(cls):
        """Check if test FMU exists for the host platform"""
        fmu_name = _PLATFORM_FMU_NAMES.get(platform.system())
        if fmu_name is None:
            raise unittest.SkipTest(f"No test FMU available for platform '{platform.system()}'")

        cls.fmu_path = TEST_DATA_DIR / fmu_name
        if not cls.fmu_path.exists():
            raise unittest.SkipTest(f"Test FMU not found: {cls.fmu_path}")

    def setUp(self):
        from pathsim.utils.fmuwrapper import FMUWrapper
        self.FMUWrapper = FMUWrapper

    def test_init_cosimulation(self):
        """Test FMUWrapper initialization for co-simulation"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')

        self.assertEqual(wrapper.mode, 'cosimulation')
        self.assertIsNotNone(wrapper.model_description)
        self.assertIsNotNone(wrapper.fmu)

    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError"""
        with self.assertRaises(ValueError) as context:
            self.FMUWrapper(str(self.fmu_path), mode='invalid_mode')

        self.assertIn("Invalid mode", str(context.exception))

    def test_fmi_version_detection(self):
        """Test FMI version is correctly detected"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')

        self.assertTrue(wrapper.fmi_version.startswith('2.') or
                       wrapper.fmi_version.startswith('3.'))

    def test_variable_maps(self):
        """Test that input/output variable maps are built"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')

        self.assertIsInstance(wrapper.input_refs, dict)
        self.assertIsInstance(wrapper.output_refs, dict)
        self.assertIsInstance(wrapper.variable_map, dict)

    def test_create_port_registers(self):
        """Test port register creation"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')
        wrapper.initialize(start_time=0.0)

        inputs, outputs = wrapper.create_port_registers()

        self.assertEqual(len(inputs), len(wrapper.input_refs))
        self.assertEqual(len(outputs), len(wrapper.output_refs))

    def test_initialize_and_step(self):
        """Test FMU initialization and stepping"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')
        wrapper.initialize(start_time=0.0)

        # Perform a step
        result = wrapper.do_step(current_time=0.0, step_size=0.01)

        self.assertFalse(result.terminate_simulation)

    def test_default_step_size(self):
        """Test default step size property"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')

        # May be None if not defined in FMU
        step_size = wrapper.default_step_size
        self.assertTrue(step_size is None or isinstance(step_size, float))

    def test_reset(self):
        """Test FMU reset"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='cosimulation')
        wrapper.initialize(start_time=0.0)

        # Step forward
        wrapper.do_step(current_time=0.0, step_size=0.01)
        wrapper.do_step(current_time=0.01, step_size=0.01)

        # Reset and reinitialize
        wrapper.reset()
        wrapper.initialize(start_time=0.0)

        # Should be able to step again from start
        result = wrapper.do_step(current_time=0.0, step_size=0.01)
        self.assertFalse(result.terminate_simulation)


@unittest.skipIf(not fmpy_available(), "FMPy not installed")
@unittest.skipIf(os.getenv("CI") == "true", "FMU tests require platform-specific binaries")
class TestFMUWrapperModelExchange(unittest.TestCase):
    """Test FMUWrapper with Model Exchange FMU"""

    @classmethod
    def setUpClass(cls):
        """Check if test FMU exists"""
        cls.fmu_path = TEST_DATA_DIR / "BouncingBall_ME.fmu"
        if not cls.fmu_path.exists():
            raise unittest.SkipTest(f"Test FMU not found: {cls.fmu_path}")

    def setUp(self):
        from pathsim.utils.fmuwrapper import FMUWrapper
        self.FMUWrapper = FMUWrapper

    def test_init_model_exchange(self):
        """Test FMUWrapper initialization for model exchange"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')

        self.assertEqual(wrapper.mode, 'model_exchange')
        self.assertIsNotNone(wrapper.model_description)
        self.assertIsNotNone(wrapper.fmu)

    def test_n_states(self):
        """Test number of continuous states"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')

        # BouncingBall has 2 states: height and velocity
        self.assertEqual(wrapper.n_states, 2)

    def test_n_event_indicators(self):
        """Test number of event indicators"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')

        # BouncingBall has 1 event indicator (ground contact)
        self.assertGreaterEqual(wrapper.n_event_indicators, 1)

    def test_initialize_model_exchange(self):
        """Test Model Exchange FMU initialization"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        event_info = wrapper.initialize(start_time=0.0, tolerance=1e-6)

        # event_info may be None for FMI 2.0
        if event_info is not None:
            self.assertFalse(event_info.terminate_simulation)

    def test_enter_continuous_time_mode(self):
        """Test entering continuous time mode"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)

        # Should not raise
        wrapper.enter_continuous_time_mode()

    def test_get_continuous_states(self):
        """Test getting continuous states"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        states = wrapper.get_continuous_states()

        self.assertIsInstance(states, np.ndarray)
        self.assertEqual(len(states), wrapper.n_states)

    def test_set_continuous_states(self):
        """Test setting continuous states"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        new_states = np.array([5.0, 0.0])  # height=5, velocity=0
        wrapper.set_continuous_states(new_states)

        states = wrapper.get_continuous_states()
        np.testing.assert_array_almost_equal(states, new_states)

    def test_get_derivatives(self):
        """Test getting state derivatives"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        wrapper.set_time(0.0)
        derivatives = wrapper.get_derivatives()

        self.assertIsInstance(derivatives, np.ndarray)
        self.assertEqual(len(derivatives), wrapper.n_states)

    def test_get_event_indicators(self):
        """Test getting event indicators"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        wrapper.set_time(0.0)
        indicators = wrapper.get_event_indicators()

        self.assertIsInstance(indicators, np.ndarray)
        self.assertEqual(len(indicators), wrapper.n_event_indicators)

    def test_event_mode_cycle(self):
        """Test entering and exiting event mode"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        # Enter event mode
        wrapper.enter_event_mode()

        # Update discrete states
        event_info = wrapper.update_discrete_states()

        self.assertFalse(event_info.terminate_simulation)

        # Return to continuous time mode
        wrapper.enter_continuous_time_mode()

    def test_completed_integrator_step(self):
        """Test completed integrator step notification"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0, tolerance=1e-6)
        wrapper.enter_continuous_time_mode()

        wrapper.set_time(0.0)

        enter_event_mode, terminate = wrapper.completed_integrator_step()

        self.assertIsInstance(enter_event_mode, bool)
        self.assertIsInstance(terminate, bool)

    def test_needs_completed_integrator_step(self):
        """Test needs_completed_integrator_step property"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')

        # Should be a boolean
        self.assertIsInstance(wrapper.needs_completed_integrator_step, bool)

    def test_provides_jacobian(self):
        """Test provides_jacobian property"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')

        # Should be a boolean
        self.assertIsInstance(wrapper.provides_jacobian, bool)

    def test_model_exchange_mode_errors_on_cosim_methods(self):
        """Test that co-simulation methods raise errors in model exchange mode"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_time=0.0)

        with self.assertRaises(RuntimeError):
            wrapper.do_step(0.0, 0.01)

    def test_set_variable(self):
        """Test setting a variable by name"""
        wrapper = self.FMUWrapper(str(self.fmu_path), mode='model_exchange')
        wrapper.initialize(start_values={"e": 0.8}, start_time=0.0, tolerance=1e-6)

        # Setting start values should work during initialization
        # After init, we can verify the FMU was configured


class TestEventInfo(unittest.TestCase):
    """Test EventInfo dataclass (no FMU required)"""

    def test_event_info_defaults(self):
        """Test EventInfo default values"""
        from pathsim.utils.fmuwrapper import EventInfo

        info = EventInfo()

        self.assertFalse(info.discrete_states_need_update)
        self.assertFalse(info.terminate_simulation)
        self.assertFalse(info.nominals_changed)
        self.assertFalse(info.values_changed)
        self.assertFalse(info.next_event_time_defined)
        self.assertEqual(info.next_event_time, 0.0)

    def test_event_info_custom_values(self):
        """Test EventInfo with custom values"""
        from pathsim.utils.fmuwrapper import EventInfo

        info = EventInfo(
            discrete_states_need_update=True,
            terminate_simulation=False,
            nominals_changed=True,
            values_changed=True,
            next_event_time_defined=True,
            next_event_time=1.5
        )

        self.assertTrue(info.discrete_states_need_update)
        self.assertFalse(info.terminate_simulation)
        self.assertTrue(info.nominals_changed)
        self.assertTrue(info.values_changed)
        self.assertTrue(info.next_event_time_defined)
        self.assertEqual(info.next_event_time, 1.5)


class TestStepResult(unittest.TestCase):
    """Test StepResult dataclass (no FMU required)"""

    def test_step_result_defaults(self):
        """Test StepResult default values"""
        from pathsim.utils.fmuwrapper import StepResult

        result = StepResult()

        self.assertFalse(result.event_encountered)
        self.assertFalse(result.terminate_simulation)
        self.assertFalse(result.early_return)
        self.assertEqual(result.last_successful_time, 0.0)

    def test_step_result_custom_values(self):
        """Test StepResult with custom values"""
        from pathsim.utils.fmuwrapper import StepResult

        result = StepResult(
            event_encountered=True,
            terminate_simulation=False,
            early_return=True,
            last_successful_time=2.5
        )

        self.assertTrue(result.event_encountered)
        self.assertFalse(result.terminate_simulation)
        self.assertTrue(result.early_return)
        self.assertEqual(result.last_successful_time, 2.5)


# RUN TESTS LOCALLY ====================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
