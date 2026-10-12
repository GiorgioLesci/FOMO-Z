"""
Unit tests.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from fomoz.binary_classification.linear_classification import LinearClassification
from fomoz.binary_classification.grid_classification import GridClassification
from fomoz.binary_classification.classifier import Classifier
from fomoz.probability.point_probability import PointProbability
from fomoz.probability.probability_function import ProbabilityFunction
from fomoz.data.dataset import Dataset
from fomoz.data.selection_definition import SelectionDefinition
from fomoz.reconstruction import Reconstruction
from fomoz.auxiliary import (
    MeasureLinear,
    build_nested_dict,
    float_representer,
    interpolate_parameters,
    make_yaml_serializable,
    restructure_dict,
)


# Shared fixture for reference dataset
@pytest.fixture
def reference_dataset():
    """Load the reference.csv dataset for testing."""
    data_path = "data/input/reference.csv"

    if not os.path.exists(data_path):
        data_path = "../data/input/reference.csv"

    if not os.path.exists(data_path):
        pytest.skip(f"Reference dataset not found at {data_path}")

    df = pd.read_csv(data_path, comment="#")
    df["weights"] = 1
    return df


# Shared fixture for dataset instance
@pytest.fixture
def reference_dataset_instance(reference_dataset):
    """Create a Dataset instance with reference.csv data for testing."""
    bands = ["g", "r", "i"]
    dataset = Dataset(reference_dataset, bands)
    return dataset


@pytest.fixture
def reference_dataset_instance_weights(reference_dataset):
    """Create a Dataset instance with reference.csv data for testing."""
    bands = ["g", "r", "i"]
    dataset = Dataset(
        reference_dataset, bands, weights=np.full(len(reference_dataset), 0.1)
    )
    return dataset


# Shared fixture for reference color list
@pytest.fixture
def reference_colors(reference_dataset):
    """Create a Dataset instance with reference.csv data for testing."""
    bands = ["g", "r", "i"]
    dataset = Dataset(reference_dataset, bands)
    return dataset.colors


# Shared fixture for target_definition instance
@pytest.fixture
def reference_target_definition_instance():
    """Create a SelectionDefinition instance."""
    z_lim = [0.5, 0.6]
    target_definition = SelectionDefinition(
        z_lim,
        z_lim_opposite=2.5,
        additional_rules=[("g", lambda g: g > 23)],
    )
    return target_definition


# Shared fixture for measure instance
@pytest.fixture
def reference_measure_instance(
    reference_dataset, reference_colors, reference_target_definition_instance
):
    """Create a MeasureLinear instance with reference.csv data for testing."""
    measure = MeasureLinear(
        reference_dataset,
        reference_colors,
        reference_target_definition_instance,
        "purity",
        0,
        0.05,
        np.linspace(0.3, 1.7, 30),
        0.05,
        0.0001,
    )
    for z_l in reference_target_definition_instance.z_lim:
        measure.process(z_l, None, None, "redshift", None, cpu_number=1)
    return measure


# Shared fixtures for calibration instances
@pytest.fixture
def reference_calibration_linear_instance(
    reference_dataset_instance, reference_target_definition_instance
):
    """Create an LinearClassification instance with reference.csv data for testing."""
    thresholded_quantity_min = 0.9
    calibration = LinearClassification(
        reference_dataset_instance,
        reference_target_definition_instance,
        thresholded_quantity_min,
        s_samples=np.linspace(0.3, 1.7, 30),
        maximized_quantity_min=0,
        step=0.0001,
        k=3,
    )

    return calibration


@pytest.fixture
def reference_calibration_linear_instance_weights(
    reference_dataset_instance_weights, reference_target_definition_instance
):
    """Create an LinearClassification instance with reference.csv data for testing."""
    thresholded_quantity_min = 0.9
    calibration = LinearClassification(
        reference_dataset_instance_weights,
        reference_target_definition_instance,
        thresholded_quantity_min,
        s_samples=np.linspace(0.3, 1.7, 30),
        maximized_quantity_min=0,
        step=0.0001,
        k=3,
    )

    return calibration


@pytest.fixture
def reference_calibration_grid_instance(
    reference_dataset_instance, reference_target_definition_instance
):
    """Create an GridClassification instance with reference.csv data for testing."""
    thresholded_quantity_min = 0.9
    calibration = GridClassification(
        reference_dataset_instance,
        reference_target_definition_instance,
        thresholded_quantity_min,
        step=0.0001,
        alpha=0.1,
    )

    return calibration


@pytest.fixture
def reference_calibration_grid_instance_weights(
    reference_dataset_instance_weights, reference_target_definition_instance
):
    """Create an GridClassification instance with reference.csv data for testing."""
    thresholded_quantity_min = 0.9
    calibration = GridClassification(
        reference_dataset_instance_weights,
        reference_target_definition_instance,
        thresholded_quantity_min,
        step=0.0001,
        alpha=0.1,
    )

    return calibration

@pytest.fixture(params=[False, True], ids=["unweighted", "weighted"])
def probability_dataset_instance(request):
    """Create a small dataset with identical colors and known redshifts."""
    n_objects = 40
    bands = ["g", "r", "i"]

    df = pd.DataFrame(
        {
            "redshift": np.linspace(0.025, 1.975, n_objects),
            "g": np.full(n_objects, 23.0),
            "r": np.full(n_objects, 22.5),
            "i": np.full(n_objects, 22.0),
            "g_error": np.full(n_objects, 0.05),
            "r_error": np.full(n_objects, 0.05),
            "i_error": np.full(n_objects, 0.05),
        }
    )

    weights = (
        np.linspace(1.0, 3.0, n_objects)
        if request.param
        else np.ones(n_objects)
    )

    return Dataset(df, bands, weights=weights)


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
                "value": 1,
            },
            (("g", "r"), ("r", "i"), 1): {
                "c": 0.3,
                "s": 0.8,
                "completeness": 0.7,
                "value": 2,
            },
            (("g", "r"), ("i", "z"), 0): {
                "c": 0.7,
                "s": 1.5,
                "completeness": 0.9,
                "value": 3,
            },
            (("r", "i"), ("i", "z"), 0): {
                "c": 0.2,
                "s": 1.1,
                "completeness": 0.6,
                "value": 4,
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
                        {"condition_index": 1, "c": -0.2},
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
        input_dict = {"g-r>c": {"z_l": [1.0, 1.5, 2.0], "c": [0.1, 0.2, 0.3]}}

        interpolate_parameters(input_dict)

        # Check that keys were renamed
        assert "c" not in input_dict["g-r>c"]  # Old key should be gone

        # Check that interpolation parameters were added
        assert "c_coefficients" in input_dict["g-r>c"]
        assert "knots" in input_dict["g-r>c"]
        assert "order" in input_dict["g-r>c"]
        assert "z_l" not in input_dict["g-r>c"]  # Should be removed

    def test_interpolate_parameters_single_point(self):
        """Test interpolation with single point (should skip interpolation)."""
        input_dict = {"g-r>c": {"z_l": [1.0], "c": [0.1]}}

        interpolate_parameters(input_dict)

        # Should rename keys but not add interpolation parameters for single point
        assert "c" not in input_dict["g-r>c"]  # Old key should be gone
        assert (
            "c_coefficients" in input_dict["g-r>c"]
        )  # No interpolation for single point
        assert "knots" in input_dict["g-r>c"]
        assert "z_l" not in input_dict["g-r>c"]  # Should be removed


class TestMakeYamlSerializable:
    """Test cases for make_yaml_serializable function."""

    def test_make_yaml_serializable_numpy_array(self):
        """Test conversion of numpy arrays to lists."""
        data = {
            "array": np.array([1, 2, 3]),
            "nested": {"float_array": np.array([1.1, 2.2, 3.3])},
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
            "regular_float": 2.71,
        }

        result = make_yaml_serializable(data)

        assert isinstance(result["int32"], int)
        assert isinstance(result["float64"], float)
        assert result["int32"] == 42
        assert result["float64"] == 3.14

    def test_make_yaml_serializable_preserves_structure(self):
        """Test that nested structure is preserved."""
        original = {
            "level1": {"level2": [1, 2, 3], "level2_dict": {"a": 1, "b": 2}},
            "simple_list": [4, 5, 6],
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
        assert hasattr(result, "value")
        assert result.tag == "tag:yaml.org,2002:float"
        # The value should be formatted with 6-digit precision
        assert "3.14159" in result.value


##################
### measure.py ###
##################


class TestMeasureProcess:
    """Test cases for Measure.process() method with reference.csv data."""

    def test_process_with_reference_data(self, reference_measure_instance):
        """Test process() method with actual reference data."""

        measure = reference_measure_instance
        z_lim = measure.td.z_lim

        # Top-level structure: one dict entry per z_l
        assert len(measure.dict_all) == len(z_lim)

        expected_pairs = [("g", "r"), ("g", "i"), ("r", "i")]  # mag1-mag2 tested pairs

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

    def test_process_with_single_z_limit(self, reference_dataset, reference_colors):
        """Test process() with single redshift limit."""
        bands = ["g", "r", "i"]
        z_l = [0.3]  # Single z limit

        # Create a new Measure instance with single z limit
        td = SelectionDefinition(z_l)
        measure = MeasureLinear(
            reference_dataset,
            reference_colors,
            td,
            "purity",
            0,
            0.05,
            np.linspace(0.3, 1.7, 30),
            0.05,
            0.0001,
        )
        measure.process(td.z_lim[0], None, None, "redshift", None, cpu_number=1)

        assert measure.td.z_lim[0] in measure.dict_all
        assert len(measure.dict_all) == 1


###################
### calibration_*.py ###
###################


class TestCalibrationProcess:
    """Test cases for the process() method with reference.csv."""

    def test_process(
        self, reference_calibration_linear_instance, reference_calibration_grid_instance
    ):
        """Test process methods with reference data and check C, P, F values."""

        # LinearClassification
        calibration = reference_calibration_linear_instance
        calibration.process(cpu_number=1)

        assert len(calibration.C) == len(calibration.td.z_lim)
        assert len(calibration.P) == len(calibration.td.z_lim)

        expected_C = [0.94084582, 0.94883587]
        expected_P = [0.90035861, 0.90166621]
        expected_F = [0.26037483, 0.2055968]

        assert np.allclose(
            calibration.C, expected_C
        ), f"Completeness mismatch for LinearClassification: expected {expected_C}, got {calibration.C}"

        assert np.allclose(
            calibration.P, expected_P
        ), f"Purity mismatch for LinearClassification: expected {expected_P}, got {calibration.P}"

        assert np.allclose(
            calibration.F, expected_F
        ), f"Failure rate mismatch for LinearClassification: expected {expected_F}, got {calibration.F}"

        # GridClassification
        calibration = reference_calibration_grid_instance
        calibration.process()

        assert len(calibration.C) == len(calibration.td.z_lim)
        assert len(calibration.P) == len(calibration.td.z_lim)

        expected_C = [0.91059957, 0.92152918]
        expected_P = [0.9103559, 0.90513834]
        expected_F = [0.22423025, 0.19189035]

        assert np.allclose(
            calibration.C, expected_C
        ), f"Completeness mismatch for GridClassification: expected {expected_C}, got {calibration.C}"

        assert np.allclose(
            calibration.P, expected_P
        ), f"Purity mismatch for GridClassification: expected {expected_P}, got {calibration.P}"

        assert np.allclose(
            calibration.F, expected_F
        ), f"Failure rate mismatch for GridClassification: expected {expected_F}, got {calibration.F}"

    def test_process_with_weights(
        self,
        reference_calibration_linear_instance_weights,
        reference_calibration_grid_instance_weights,
    ):
        """Test process methods with reference data and check C, P, F values."""

        # LinearClassification
        calibration = reference_calibration_linear_instance_weights
        calibration.process(cpu_number=1)

        assert len(calibration.C) == len(calibration.td.z_lim)
        assert len(calibration.P) == len(calibration.td.z_lim)

        expected_C = [0.94084582, 0.94883587]
        expected_P = [0.90035861, 0.90166621]
        expected_F = [0.26037483, 0.2055968]

        assert np.allclose(
            calibration.C, expected_C
        ), f"Completeness mismatch for LinearClassification: expected {expected_C}, got {calibration.C}"

        assert np.allclose(
            calibration.P, expected_P
        ), f"Purity mismatch for LinearClassification: expected {expected_P}, got {calibration.P}"

        assert np.allclose(
            calibration.F, expected_F
        ), f"Failure rate mismatch for LinearClassification: expected {expected_F}, got {calibration.F}"

        # GridClassification
        calibration = reference_calibration_grid_instance_weights
        calibration.process()

        assert len(calibration.C) == len(calibration.td.z_lim)
        assert len(calibration.P) == len(calibration.td.z_lim)

        expected_C = [0.91059957, 0.92152918]
        expected_P = [0.9103559, 0.90513834]
        expected_F = [0.22423025, 0.19189035]

        assert np.allclose(
            calibration.C, expected_C
        ), f"Completeness mismatch for GridClassification: expected {expected_C}, got {calibration.C}"

        assert np.allclose(
            calibration.P, expected_P
        ), f"Purity mismatch for GridClassification: expected {expected_P}, got {calibration.P}"

        assert np.allclose(
            calibration.F, expected_F
        ), f"Failure rate mismatch for GridClassification: expected {expected_F}, got {calibration.F}"


############################
### classifier.py ###
############################


class TestColorSelection:
    """Test cases for Classifier.process() method with reference.csv."""

    def test_process(
        self,
        reference_dataset,
        reference_calibration_linear_instance,
        reference_calibration_grid_instance,
    ):
        """Test process method with reference data and check the number of selected objects."""
        calibration = reference_calibration_linear_instance
        calibration.process(cpu_number=1)

        calibration_grid = reference_calibration_grid_instance
        calibration_grid.process()

        # "Test" sample
        test_sample = reference_dataset

        # Select - linear
        zl_new = np.arange(min(calibration.td.z_lim), max(calibration.td.z_lim), 0.01)

        sel = Classifier()
        sel.load_parameters(calibration)
        sel.process(zl_new, test_sample)
        mask = sel.mask

        expected_N = [3904, 3872, 3847, 3822, 3798, 3776, 3762, 3748, 3723, 3700, 3661]

        assert len(expected_N) == len(mask)

        for i in range(len(zl_new)):
            assert np.allclose(
                np.sum(mask[i]), expected_N[i]
            ), f"[Linear calibration]: Number of selected galaxies mismatch for the {i}-th test z_l: expected {expected_N[i]}, got {np.sum(mask[i])}"

        # Select - grid
        zl_new = [
            calibration_grid.td.z_lim[0] - 0.05,
            calibration_grid.td.z_lim[0],
            calibration_grid.td.z_lim[0] + 0.05,
            calibration_grid.td.z_lim[1],
            calibration_grid.td.z_lim[1] + 0.05,
        ]

        sel = Classifier()
        sel.load_parameters(calibration_grid)
        sel.process(zl_new, test_sample)
        mask = sel.mask

        expected_N = [3737, 3737, 3542, 3542, 0]

        assert len(expected_N) == len(mask)

        for i in range(len(zl_new)):
            assert np.allclose(
                np.sum(mask[i]), expected_N[i]
            ), f"[Grid calibration]: Number of selected galaxies mismatch for the {i}-th test z_l: expected {expected_N[i]}, got {np.sum(mask[i])}"


###########################
### point_probability.py ###
###########################


class TestPointProbability:
    """Test point probabilities against known weighted memberships."""

    def test_process(self, probability_dataset_instance):
        """Test probabilities when all reference objects match."""
        dataset = probability_dataset_instance
        td = SelectionDefinition(0.5)

        point_prob = PointProbability(td, dataset, verbose=False)
        point_prob.process(cpu_number=1)

        probabilities = point_prob.probability_dictionary["prob"]

        redshift = dataset.dataset["redshift"].to_numpy()
        weights = dataset.dataset["weights"].to_numpy()

        expected_probability = (
            np.sum(weights[redshift > 0.5]) / np.sum(weights)
        )

        assert probabilities.shape == (len(dataset.dataset),)
        assert np.all(np.isfinite(probabilities))
        assert np.allclose(
            probabilities, expected_probability
        ), (
            f"Probability mismatch: expected {expected_probability}, "
            f"got {probabilities}"
        )

        # Check that exporting results preserves row alignment.
        results = point_prob.get_results_dataframe()

        pd.testing.assert_frame_equal(
            results[dataset.dataset.columns],
            dataset.dataset,
        )


##############################
### probability_function.py ###
##############################


class TestProbabilityFunction:
    """Test cumulative estimates, PDF normalization, and refitting."""

    def test_process_and_update_fit(self, probability_dataset_instance):
        """Test reconstruction against a known weighted cumulative."""
        dataset = probability_dataset_instance

        prob_func = ProbabilityFunction(
            dataset,
            delta=0.1,
            min_variable=0.01,
            max_variable=2.1,
            verbose=False,
        )

        prob_func.process(
            cpu_number=1,
            n_internal_knots=3,
        )

        redshift = dataset.dataset["redshift"].to_numpy()
        weights = dataset.dataset["weights"].to_numpy()
        x = prob_func.variable_samples
        n_objects = len(dataset.dataset)

        expected_cumulative = np.array(
            [
                np.sum(weights[redshift < threshold]) / np.sum(weights)
                for threshold in x
            ]
        )

        assert prob_func.cumulative_matrix.shape == (
            len(x), n_objects
        )
        assert np.allclose(
            prob_func.cumulative_matrix,
            expected_cumulative[:, None],
        ), "Cumulative probabilities differ from the expected weighted fractions."

        assert np.all(prob_func.fit_success)
        assert prob_func.probability_functions.shape == (
            n_objects, len(x)
        )
        assert np.all(np.isfinite(prob_func.probability_functions))
        assert np.all(prob_func.probability_functions >= 0.0)

        integrals = np.trapz(
            prob_func.probability_functions,
            x=x,
            axis=1,
        )
        assert np.allclose(integrals, 1.0)

        # Check the ordering of the reconstructed percentiles.
        results = prob_func.get_results_dataframe()

        assert np.all(results["fomoz_p16"] <= results["fomoz_median"])
        assert np.all(results["fomoz_median"] <= results["fomoz_p84"])

        pd.testing.assert_frame_equal(
            results[dataset.dataset.columns],
            dataset.dataset,
        )

        # Refit without changing the previously computed cumulative estimates.
        cumulative_before = prob_func.cumulative_matrix.copy()

        prob_func.update_fit(n_internal_knots=5)

        np.testing.assert_array_equal(
            prob_func.cumulative_matrix,
            cumulative_before,
        )

        assert prob_func.n_internal_knots == 5
        assert np.all(prob_func.fit_success)
        assert np.all(np.isfinite(prob_func.probability_functions))
        assert np.all(prob_func.probability_functions >= 0.0)

        integrals = np.trapz(
            prob_func.probability_functions,
            x=x,
            axis=1,
        )
        assert np.allclose(integrals, 1.0)


#########################
### reconstruction.py ###
#########################


class TestReconstruction:
    """Test reconstructed histograms against known weighted counts."""

    def test_get_distribution(self, probability_dataset_instance):
        """Test weighted counts and normalized probability densities."""
        dataset = probability_dataset_instance
        td = SelectionDefinition(0.5)

        point_prob = PointProbability(td, dataset, verbose=False)
        point_prob.process(cpu_number=1)

        reconstruction = Reconstruction(point_prob)

        redshift = dataset.dataset["redshift"].to_numpy()
        weights = dataset.dataset["weights"].to_numpy()
        bin_edges = np.array([0.0, 0.4, 1.0, 2.0])

        # Identical colors imply the same probability for every object.
        expected_probability = (
            np.sum(weights[redshift > 0.5]) / np.sum(weights)
        )

        expected_counts, _ = np.histogram(
            redshift,
            bins=bin_edges,
            weights=weights * expected_probability,
        )

        counts = reconstruction.get_distribution(
            "redshift",
            bin_edges,
            normalize=False,
        )

        assert counts.shape == (len(bin_edges) - 1,)
        assert np.allclose(
            counts, expected_counts
        ), (
            f"Reconstructed counts mismatch: expected {expected_counts}, "
            f"got {counts}"
        )

        # Unequal bin widths also test conversion from counts to density.
        expected_density = (
            expected_counts / np.sum(expected_counts) / np.diff(bin_edges)
        )

        density = reconstruction.get_distribution(
            "redshift",
            bin_edges,
            normalize=True,
        )

        assert np.allclose(
            density, expected_density
        ), (
            f"Reconstructed density mismatch: expected {expected_density}, "
            f"got {density}"
        )

        assert np.isclose(
            np.sum(density * np.diff(bin_edges)),
            1.0,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
