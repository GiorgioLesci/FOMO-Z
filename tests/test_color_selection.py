"""
Unit tests for the color_selection subpackage.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import pytest
import numpy as np
import yaml
import pandas as pd
import os

from fomoz.color_selection.auxiliary import (
    compute_metrics,
    compute_for_simple,
    compute_for_cross,
    build_nested_dict,
    restructure_dict,
    interpolate_parameters,
    make_yaml_serializable,
    float_representer
)
from fomoz.color_selection.measure import Measure
from fomoz.color_selection.optimize import Optimize
from fomoz.color_selection.select_galaxy import SelectGalaxy


# Shared fixture for reference dataset
@pytest.fixture
def reference_dataset():
    """Load the reference.csv dataset for testing."""
    data_path = "data/input/reference.csv"
    
    if not os.path.exists(data_path):
        data_path = "../data/input/reference.csv"
    
    if not os.path.exists(data_path):
        pytest.skip(f"Reference dataset not found at {data_path}")
    
    return pd.read_csv(data_path, comment="#")


# Shared fixture for measure instance
@pytest.fixture
def reference_measure_instance(reference_dataset):
    """Create a Measure instance with reference.csv data for testing."""
    bands = ['g', 'r', 'i']
    z_lim = [0.5, 0.6]
    
    measure = Measure(reference_dataset, bands, z_lim, initial_threshold=0.8)
    measure.process(cpu_number=1)
    
    return measure


# Shared fixture for optimize instance  
@pytest.fixture
def reference_optimize_instance(reference_measure_instance):
    """Create an Optimize instance with reference.csv data for testing."""
    optimize = Optimize(reference_measure_instance)
    
    return optimize


####################
### auxiliary.py ###
####################


class TestBuildNestedDict:
    """Test cases for build_nested_dict function."""
    
    def test_build_nested_dict(self):
        """Test building nested dictionary from cut dictionary."""
        cut_dict = {
            (("g", "r"), ("r", "i"), 0): {
                "c": 0.5, 
                "s": 1.2, 
                "completeness": 0.8,
                "value": 1
            },
            (("g", "r"), ("r", "i"), 1): {
                "c": 0.3, 
                "s": 0.8, 
                "completeness": 0.7,
                "value": 2
            },
            (("g", "r"), ("i", "z"), 0): {
                "c": 0.7, 
                "s": 1.5, 
                "completeness": 0.9,
                "value": 3
            },
            (("r", "i"), ("i", "z"), 0): {
                "c": 0.2, 
                "s": 1.1, 
                "completeness": 0.6,
                "value": 4
            },
        }
        
        result = build_nested_dict(cut_dict)
        
        # Check top-level structure
        assert ("g", "r") in result
        assert ("r", "i") in result
        
        # Check nested structure for ("g", "r")
        assert ("r", "i") in result["g", "r"]
        assert ("i", "z") in result["g", "r"]
        
        # Check number of cuts in each comp_pair
        assert len(result["g", "r"]["r", "i"]) == 2
        assert len(result["g", "r"]["i", "z"]) == 1
        assert len(result["r", "i"]["i", "z"]) == 1
        
        # Check that condition_index and parameters are preserved
        cut_1 = result["g", "r"]["r", "i"][0]
        assert "condition_index" in cut_1
        assert "c" in cut_1
        assert "s" in cut_1
        assert "completeness" in cut_1


class TestRestructureDict:
    """Test cases for restructure_dict function."""
    
    def test_restructure_dict_simple_cut(self):
        """Test restructuring with simple color cuts."""
        input_dict = {
            1.0: {
                ("g", "r"): {
                    ("g", "r"): [
                        {"condition_index": 0, "c": 0.5},
                        {"condition_index": 1, "c": -0.2}
                    ]
                }
            }
        }
        
        bands = ["g", "r", "i"]
        colors = [("g", "r")]
        z_l = [1.0]
        purity_min = [0.8]
        
        restructure_dict(input_dict, bands, colors, z_l, "purity", purity_min)
        
        assert "g-r>c" in input_dict
        assert "g-r<c" in input_dict
        assert input_dict["g-r>c"]["c"] == [0.5]
        assert input_dict["g-r<c"]["c"] == [-0.2]
        assert "metadata" in input_dict
        assert input_dict["metadata"]["bands"] == bands


class TestInterpolateParameters:
    """Test cases for interpolate_parameters function."""
    
    def test_interpolate_parameters_basic(self):
        """Test parameter interpolation with multiple points."""
        input_dict = {
            "g-r>c": {
                "z_l": [1.0, 1.5, 2.0],
                "c": [0.1, 0.2, 0.3]
            }
        }
        
        interpolate_parameters(input_dict)
        
        # Check that keys were renamed
        assert "c" not in input_dict["g-r>c"]  # Old key should be gone
        assert "c_optimal" in input_dict["g-r>c"]  # New key should exist
        
        # Check that interpolation parameters were added
        assert "c_coefficients" in input_dict["g-r>c"]
        assert "knots" in input_dict["g-r>c"]
        assert "order" in input_dict["g-r>c"]
        assert "z_l" not in input_dict["g-r>c"]  # Should be removed
    
    def test_interpolate_parameters_single_point(self):
        """Test interpolation with single point (should skip interpolation)."""
        input_dict = {
            "g-r>c": {
                "z_l": [1.0],
                "c": [0.1]
            }
        }
        
        interpolate_parameters(input_dict)
        
        # Should rename keys but not add interpolation parameters for single point
        assert "c" not in input_dict["g-r>c"]  # Old key should be gone
        assert "c_optimal" in input_dict["g-r>c"]  # New key should exist
        assert "c_coefficients" not in input_dict["g-r>c"]  # No interpolation for single point
        assert "knots" in input_dict["g-r>c"]
        assert "z_l" not in input_dict["g-r>c"]  # Should be removed


class TestMakeYamlSerializable:
    """Test cases for make_yaml_serializable function."""
    
    def test_make_yaml_serializable_numpy_array(self):
        """Test conversion of numpy arrays to lists."""
        data = {
            "array": np.array([1, 2, 3]),
            "nested": {
                "float_array": np.array([1.1, 2.2, 3.3])
            }
        }
        
        result = make_yaml_serializable(data)
        
        assert isinstance(result["array"], list)
        assert isinstance(result["nested"]["float_array"], list)
        assert result["array"] == [1, 2, 3]
    
    def test_make_yaml_serializable_numpy_scalars(self):
        """Test conversion of numpy scalar types."""
        data = {
            "int32": np.int32(42),
            "float64": np.float64(3.14),
            "regular_int": 100,
            "regular_float": 2.71
        }
        
        result = make_yaml_serializable(data)
        
        assert isinstance(result["int32"], int)
        assert isinstance(result["float64"], float)
        assert result["int32"] == 42
        assert result["float64"] == 3.14
    
    def test_make_yaml_serializable_preserves_structure(self):
        """Test that nested structure is preserved."""
        original = {
            "level1": {
                "level2": [1, 2, 3],
                "level2_dict": {"a": 1, "b": 2}
            },
            "simple_list": [4, 5, 6]
        }
        
        result = make_yaml_serializable(original)
        
        assert result["level1"]["level2"] == [1, 2, 3]
        assert result["level1"]["level2_dict"]["a"] == 1
        assert result["simple_list"] == [4, 5, 6]


class TestFloatRepresenter:
    """Test cases for float_representer function."""
    
    def test_float_representer_zero(self):
        """Test representation of zero value."""
        dumper = yaml.Dumper(stream=None)
        result = float_representer(dumper, 0.0)
        assert result.value == "0"
        assert result.tag == "tag:yaml.org,2002:int"
    
    def test_float_representer_integer_float(self):
        """Test representation of float that is integer."""
        dumper = yaml.Dumper(stream=None)
        result = float_representer(dumper, 42.0)
        assert result.value == "42"
        assert result.tag == "tag:yaml.org,2002:int"
    
    def test_float_representer_regular_float(self):
        """Test representation of regular float."""
        dumper = yaml.Dumper(stream=None)
        result = float_representer(dumper, 3.14159)
        # Check that it returns a ScalarNode with float tag
        assert hasattr(result, 'value')
        assert result.tag == "tag:yaml.org,2002:float"
        # The value should be formatted with 6-digit precision
        assert '3.14159' in result.value


##################
### measure.py ###
##################

class TestMeasureProcess:
    """Test cases for Measure.process() method with reference.csv data."""

    def test_process_with_reference_data(self, reference_measure_instance):
        """Test process() method with actual reference data."""
        
        measure = reference_measure_instance
        z_lim = measure.z_lim

        # Top-level structure: one dict entry per z_l
        assert len(measure.dict_all) == len(z_lim)

        expected_pairs = [('g', 'r'), ('g', 'i'), ('r', 'i')]  # mag1-mag2 tested pairs

        for z_l in z_lim:
            assert z_l in measure.dict_all
            assert isinstance(measure.dict_all[z_l], dict)

            for base_pair in expected_pairs:
                assert base_pair in measure.dict_all[z_l]
                base_data = measure.dict_all[z_l][base_pair]
                assert isinstance(base_data, dict)

                # Simple cuts (base_pair compared with itself)
                assert base_pair in base_data
                simple_data = base_data[base_pair]
                assert isinstance(simple_data, dict)

                # loop both condition_index: 0 (>c) and 1 (<c)
                for cond_index in (0, 1):
                    if cond_index in simple_data:
                        cond_metrics = simple_data[cond_index]
                        assert isinstance(cond_metrics, dict)

                        # required keys exist
                        for key in ("c", "purity", "completeness"):
                            assert key in cond_metrics

                        # array lengths consistent
                        n = len(cond_metrics["c"])
                        assert len(cond_metrics["purity"]) == n
                        assert len(cond_metrics["completeness"]) == n

                # Cross-color cuts exist if defined
                for comp_pair, comp_data in base_data.items():
                    if comp_pair == base_pair:
                        continue  # already checked simple case

                    # this is a cross-color structure: dict indexed by cond_index
                    assert isinstance(comp_data, dict)
                    for cond_index, cond_metrics in comp_data.items():
                        assert isinstance(cond_index, int)
                        assert 0 <= cond_index <= 3  # 4 possible orientation cuts

                        for key in ("s", "c", "purity", "completeness"):
                            assert key in cond_metrics

                        n = len(cond_metrics["c"])
                        assert n == len(cond_metrics["s"])  # matching lengths
                        assert n == len(cond_metrics["purity"])
                        assert n == len(cond_metrics["completeness"])

    def test_process_with_single_z_limit(self, reference_dataset):
        """Test process() with single redshift limit."""
        bands = ['g', 'r', 'i']
        z_l = [0.3]  # Single z limit

        # Create a new Measure instance with single z limit
        measure = Measure(reference_dataset, bands, z_l)
        measure.process(cpu_number=1)

        assert 0.3 in measure.dict_all
        assert len(measure.dict_all) == 1


###################
### optimize.py ###
###################

class TestOptimizeProcess:
    """Test cases for the process() method with reference.csv."""
    
    def test_process(self, reference_optimize_instance):
        """Test process method with reference data and check C, P, F values."""
        optimize = reference_optimize_instance

        # Run process (iterative method)
        threshold_tot_min = 0.9
        optimize.process(
            threshold_tot_min=threshold_tot_min,
            step=0.0001,
            cpu_number=1,
            k=3
        )

        # Check that C_tot and P_tot arrays are populated correctly
        assert len(optimize.C_iterative_tot) == len(optimize.z_lim)
        assert len(optimize.P_iterative_tot) == len(optimize.z_lim)

        # Check against known expected values
        expected_C = [0.9294421,  0.93034951]
        expected_P = [0.90339485, 0.90398241]
        expected_F = [0.43983402, 0.31364725]

        assert np.allclose(optimize.C_iterative_tot, expected_C), \
            f"Completeness mismatch: expected {expected_C}, got {optimize.C_iterative_tot}"

        assert np.allclose(optimize.P_iterative_tot, expected_P), \
            f"Purity mismatch: expected {expected_P}, got {optimize.P_iterative_tot}"

        assert np.allclose(optimize.F_iterative_tot, expected_F), \
            f"Failure rate mismatch: expected {expected_F}, got {optimize.F_iterative_tot}"


############################
### select_galaxy.py ###
############################

class TestSelectGalaxy:
    """Test cases for SelectGalaxy.process() method with reference.csv."""
    
    def test_process(self, reference_dataset, reference_optimize_instance):
        """Test process method with reference data and check the number of selected objects."""
        optimize = reference_optimize_instance
        threshold_tot_min = 0.9
        optimize.process(
            threshold_tot_min=threshold_tot_min,
            step=0.0001,
            cpu_number=1,
            k=3
        )

        # "Test" sample and test z_l points
        test_sample = reference_dataset
        zl_new = np.arange(min(optimize.z_lim), max(optimize.z_lim), 0.01)

        # Select
        sel = SelectGalaxy()
        sel.load_parameters(optimize)
        sel.process(zl_new, test_sample)
        mask = sel.mask
        
        # Check the number of selected objects
        expected_N = np.zeros(len(mask))
        expected_N = [4389, 4353, 4327, 4302, 4277, 4246, 4213, 4186, 4155, 4122]

        for i in range(len(zl_new)):
            assert np.allclose(np.sum(mask[i]), expected_N[i]), \
                f"Number of selected galaxies mismatch for the {i}-th test z_l: expected {expected_N[i]}, got {np.sum(mask[i])}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
