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
from fomoz.color_selection.select_background import SelectBackground


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
    
    measure = Measure(reference_dataset, bands, z_lim)
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

class TestComputeMetrics:
    """Test cases for compute_metrics function."""
    
    def test_compute_metrics_basic(self):
        """Test basic metric computation with clear separation."""
        mask = np.array([True, True, False, False, True, False])
        redshift = np.array([0.8, 1.2, 0.5, 1.5, 1.1, 0.9])
        z_l = 1.0
        
        TP, FP, FN, TN = compute_metrics(mask, redshift, z_l)
        
        assert TP == 2  # redshift > z_l and mask True: indices 1, 4
        assert FP == 1  # redshift < z_l and mask True: index 0
        assert FN == 1  # redshift > z_l and mask False: index 3
        assert TN == 2  # redshift < z_l and mask False: indices 2, 5
    
    def test_compute_metrics_all_true(self):
        """Test when all objects are selected."""
        mask = np.array([True, True, True])
        redshift = np.array([1.1, 0.9, 1.2])
        z_l = 1.0
        
        TP, FP, FN, TN = compute_metrics(mask, redshift, z_l)
        
        assert TP == 2
        assert FP == 1
        assert FN == 0
        assert TN == 0
    
    def test_compute_metrics_all_false(self):
        """Test when no objects are selected."""
        mask = np.array([False, False, False])
        redshift = np.array([1.1, 0.9, 1.2])
        z_l = 1.0
        
        TP, FP, FN, TN = compute_metrics(mask, redshift, z_l)
        
        assert TP == 0
        assert FP == 0
        assert FN == 2
        assert TN == 1


class TestComputeForSimple:
    """Test cases for compute_for_simple function."""
    
    def test_compute_for_simple_basic(self):
        """Test simple threshold computation."""
        c = 0.5
        color = np.array([0.6, 0.4, 0.7, 0.3])
        redshift = np.array([1.1, 0.9, 1.2, 0.8])
        z_l = 1.0
        
        args = (c, color, redshift, z_l)
        result = compute_for_simple(args)
        
        # Expected: [TP_up, FP_up, FN_up, TN_up, TP_low, FP_low, FN_low, TN_low]
        assert len(result) == 8
        assert result == [2, 0, 0, 2, 0, 2, 2, 0]  # Verify based on input data


class TestComputeForCross:
    """Test cases for compute_for_cross function."""
    
    def test_compute_for_cross_basic(self):
        """Test cross-color metric computation."""
        s, c = 1.0, 0.0
        color1 = np.array([1.0, 0.5, 1.5, 0.0])
        color2 = np.array([0.5, 0.5, 1.0, 0.0])
        redshift = np.array([1.1, 0.9, 1.2, 0.8])
        z_l = 1.0
        
        args = (s, c, color1, color2, redshift, z_l)
        results = compute_for_cross(args)
        
        assert len(results) == 4
        for result in results:
            assert len(result) == 4  # Each should be (TP, FP, FN, TN)


class TestBuildNestedDict:
    """Test cases for build_nested_dict function."""
    
    def test_build_nested_dict_basic(self):
        """Test building nested dictionary from cut list."""
        cut_list = [
            {"base_pair": ("g", "r"), "comp_pair": ("r", "i"), "value": 1},
            {"base_pair": ("g", "r"), "comp_pair": ("r", "i"), "value": 2},
            {"base_pair": ("g", "r"), "comp_pair": ("i", "z"), "value": 3},
            {"base_pair": ("r", "i"), "comp_pair": ("i", "z"), "value": 4},
        ]
        
        result = build_nested_dict(cut_list)
        
        assert ("g", "r") in result
        assert ("r", "i") in result["g", "r"]
        assert ("i", "z") in result["g", "r"]
        assert len(result["g", "r"]["r", "i"]) == 2
        assert len(result["g", "r"]["i", "z"]) == 1


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
        
        restructure_dict(input_dict, bands, colors, z_l, purity_min)
        
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
        assert "knots" not in input_dict["g-r>c"]
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
        # Initialize Measure with reference data
        measure = reference_measure_instance
        z_lim = measure.z_lim
        
        # Verify the results structure
        assert len(measure.dict_all) == len(z_lim)
        
        for z_l in z_lim:
            assert z_l in measure.dict_all
            assert isinstance(measure.dict_all[z_l], dict)
            
            # Check that all color combinations are present
            expected_colors = [('g', 'r'), ('g', 'i'), ('r', 'i')]
            
            for color_pair in expected_colors:
                assert color_pair in measure.dict_all[z_l]
                
                # Check that each color pair has results
                color_data = measure.dict_all[z_l][color_pair]
                assert isinstance(color_data, dict)
                
                # Check self-comparison (simple cuts)
                assert color_pair in color_data
                simple_cut_data = color_data[color_pair]
                assert 'c' in simple_cut_data
                assert 'TP' in simple_cut_data
                assert 'FP' in simple_cut_data
                assert 'FN' in simple_cut_data
                assert 'TN' in simple_cut_data
                
                # Verify data types and non-empty results
                assert len(simple_cut_data['c']) > 0
                assert len(simple_cut_data['TP']) == len(simple_cut_data['c'])
                assert len(simple_cut_data['FP']) == len(simple_cut_data['c'])
    
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

class TestOptimizeProcessWithReference:
    """Test cases for Optimize.process_L24() and process() methods with reference.csv."""
    
    def test_process_L24_with_reference_data(self, reference_optimize_instance):
        """Test process_L24 method with reference data and check C/P values."""
        optimize = reference_optimize_instance
        
        # Run process_L24
        purity_min = 0.9
        optimize.process_L24(
            purity_min=purity_min,
            completeness_min=0.0,
            smoothing=0.0,
            k=3
        )
        
        # Check that C_tot and P_tot arrays are populated correctly
        assert len(optimize.C_single_tot) == len(optimize.z_lim)
        assert len(optimize.P_single_tot) == len(optimize.z_lim)
        
        # Check against known expected values
        expected_C = [0.96249414, 0.94920795]
        expected_P = [0.86606201, 0.86681975]
        expected_F = [0.65871369, 0.46288907]

        assert np.allclose(optimize.C_single_tot, expected_C, atol=0.05), \
            f"Completeness mismatch: expected {expected_C}, got {optimize.C_single_tot}"

        assert np.allclose(optimize.P_single_tot, expected_P, atol=0.05), \
            f"Purity mismatch: expected {expected_P}, got {optimize.P_single_tot}"

        assert np.allclose(optimize.F_single_tot, expected_F, atol=0.05), \
            f"Failure rate mismatch: expected {expected_F}, got {optimize.F_single_tot}"
    
    def test_process_with_reference_data(self, reference_optimize_instance):
        """Test process method with reference data and check C/P values."""
        optimize = reference_optimize_instance
        
        # Run process (iterative method)
        purity_tot_min = 0.9
        optimize.process(
            purity_tot_min=purity_tot_min,
            start_purity=0.999,
            step=0.0001,
            cpu_number=1,
            smoothing=0.0,
            k=3
        )
        
        # Check that C_tot and P_tot arrays are populated correctly
        assert len(optimize.C_iterative_tot) == len(optimize.z_lim)
        assert len(optimize.P_iterative_tot) == len(optimize.z_lim)
        
        # Check against known expected values
        expected_C = [0.93084857, 0.92884083]
        expected_P = [0.90172492, 0.90561412]
        expected_F = [0.44917012, 0.30726257]

        assert np.allclose(optimize.C_iterative_tot, expected_C, atol=0.05), \
            f"Completeness mismatch: expected {expected_C}, got {optimize.C_iterative_tot}"

        assert np.allclose(optimize.P_iterative_tot, expected_P, atol=0.05), \
            f"Purity mismatch: expected {expected_P}, got {optimize.P_iterative_tot}"

        assert np.allclose(optimize.F_iterative_tot, expected_F, atol=0.05), \
            f"Failure rate mismatch: expected {expected_F}, got {optimize.F_iterative_tot}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
