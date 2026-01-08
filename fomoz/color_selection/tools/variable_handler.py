"""
Data conversions and dictionary restructuring.

This module provides auxiliary functions for data conversion and
dictionary reshaping.

:Author: Giorgio Francesco Lesci <lesci.giorgio [at] gmail [dot] com>
"""

import numpy as np


def set_input_variable_precision(var, prec=6):
    """
    Convert input variable to float32 numpy array and apply precision rounding.

    Parameters
    ----------
    var : scalar, list, or numpy array
        Input variable to convert and round.
    prec : float, optional
        Decimal places for rounding (default: 6).

    Returns
    -------
    numpy.ndarray
        float32 array with values rounded to specified precision.
        Always returns at least 1D array.
    """
    return np.atleast_1d(np.asarray(var, dtype=np.float32)).round(prec)


def build_nested_dict(cuts):
    """
    Build a nested dictionary from a list of color cuts.

    This utility organizes a flat list of cuts into a two-level nested
    dictionary structure based on their associated color pairs. Each
    cut is grouped first by its `base_pair`, then by its `comp_pair`.

    Parameters
    ----------
    cuts : dict
        Dictionary of cut definitions, where keys are tuples of the form
        (base_pair, comp_pair, condition_index) and values are dictionaries
        containing cut parameters. Each value dictionary must contain:
        - "c" : float
            Intercept parameter of the cut
        - "completeness" : float
            Completeness value of the cut
        - "s" : float, optional
            Slope parameter (present only for cross-color cuts)

    Returns
    -------
    dict
        A nested dictionary of the form:
        {
            base_pair1: {
                comp_pair1: [cut1, cut2, ...],
                comp_pair2: [...],
                ...
            },
            base_pair2: {
                ...
            },
            ...
        }
    """
    nested = {}
    for key, cut in cuts.items():
        base_pair, comp_pair, cond_index = key
        nested.setdefault(base_pair, {}).setdefault(comp_pair, []).append(
            {
                "condition_index": cond_index,
                "c": cut["c"],
                "s": cut.get("s"),  # None for simple cuts
                "completeness": cut["completeness"],
            }
        )
    return nested


def restructure_dict(
    input_dict,
    bands,
    colors,
    z_lim,
    thresholded_quantity,
    threshold,
    is_zlim_lower=True,
):
    """
    Restructure the nested dictionary of color conditions.

    This method converts the input dictionary of cuts (organized by
    redshift bins, base pairs, and comparison pairs) into a flatter
    representation keyed by human-readable string expressions of the
    color conditions.

    Parameters
    ----------
    input_dict : dict
        Nested dictionary of the form::
            {
                z_l: {
                    color1: {
                        color2: [list of cut dictionaries]
                    }
                }
            }

        Each cut dictionary must include:
        - "condition_index" : int
            Index specifying which inequality is applied (0–3).
        - "c" : float
            Intercept of the cut.
        - "s" : float, optional
            Slope of the cut (present only for cross-color cuts).

    Returns
    -------
    dict
        Restructured dictionary keyed by string expressions of the cuts,
        e.g. ``"g-r > s*(r-i) + c"``. Each entry stores:
        - "z_l" : list of float
            Redshift thresholds associated with the cut.
        - "c" : list of float
            Intercepts of the cut across redshifts.
        - "s" : list of float, optional
            Slopes of the cut (if applicable).
    """
    dict_restructured = {}
    dict_restructured["metadata"] = {}
    dict_restructured["metadata"]["bands"] = bands
    dict_restructured["metadata"]["z_l"] = [float(z) for z in z_lim]
    dict_restructured["metadata"]["z_l_is_lower_limit"] = is_zlim_lower
    dict_restructured["metadata"][thresholded_quantity + "_min"] = [
        float(T) for T in threshold
    ]

    for i in range(len(colors)):
        for j in range(i, len(colors)):
            mag1, mag2 = colors[i][0], colors[i][1]
            mag3, mag4 = colors[j][0], colors[j][1]
            color1 = (mag1, mag2)
            color2 = (mag3, mag4)
            for z_l in z_lim:

                if color1 in input_dict[z_l]:

                    if color2 in input_dict[z_l][color1]:

                        dic = input_dict[z_l][color1][color2]
                        for k in range(len(dic)):

                            if color1 == color2 and dic[k]["condition_index"] == 0:
                                expr = f"{mag1}-{mag2}>c"
                            elif color1 == color2 and dic[k]["condition_index"] == 1:
                                expr = f"{mag1}-{mag2}<c"
                            elif color1 != color2 and dic[k]["condition_index"] == 0:
                                expr = f"{mag1}-{mag2}>s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 1:
                                expr = f"{mag1}-{mag2}>-s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 2:
                                expr = f"{mag1}-{mag2}<s*({mag3}-{mag4})+c"
                            elif color1 != color2 and dic[k]["condition_index"] == 3:
                                expr = f"{mag1}-{mag2}<-s*({mag3}-{mag4})+c"

                            if expr not in dict_restructured:
                                dict_restructured[expr] = {}

                            if "z_l" not in dict_restructured[expr]:
                                dict_restructured[expr]["z_l"] = []
                            dict_restructured[expr]["z_l"].append(float(z_l))

                            if "c" not in dict_restructured[expr]:
                                dict_restructured[expr]["c"] = []
                            dict_restructured[expr]["c"].append(dic[k]["c"])

                            if "s" in expr:
                                if "s" not in dict_restructured[expr]:
                                    dict_restructured[expr]["s"] = []
                                dict_restructured[expr]["s"].append(dic[k]["s"])
    input_dict.clear()
    input_dict.update(dict_restructured)


def make_yaml_serializable(obj):
    """
    Convert complex or NumPy-based data types into YAML serializables.

    This function traverses nested data structures (dictionaries,
    lists, tuples) and converts objects that are not natively
    supported by the YAML safe dumper (such as NumPy arrays or
    NumPy scalar types) into equivalent Python representations.

    Specifically:
        - ``numpy.ndarray`` is converted to ``list``
        - ``numpy.generic`` (e.g. ``numpy.float64``, ``numpy.int32``)
          is converted to corresponding Python ``float`` or ``int``
        - Nested containers (``dict``, ``list``, ``tuple``)
          are processed recursively
        - All other objects are returned unchanged

    Parameters
    ----------
    obj : any
        The input object (typically a dictionary or nested structure)
        to be converted into a YAML-safe form.

    Returns
    -------
    any
        A copy of the input object containing only built-in Python types
        (``dict``, ``list``, ``tuple``, ``str``, ``int``, ``float``,
        ``bool``, ``None``), suitable for serialization.
    """
    if isinstance(obj, dict):
        return {k: make_yaml_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_yaml_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(make_yaml_serializable(v) for v in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # catches np.float32, np.int64, etc.
        return obj.item()
    else:
        return obj


def float_representer(dumper, value):
    """
    Get custom YAML representer for floating-point numbers.

    This function defines how floating-point values are converted to text
    when dumping data to a YAML file using ``yaml.safe_dump``.

    Parameters
    ----------
    dumper : yaml.Dumper
        The YAML dumper instance provided by PyYAML during serialization.
    value : float
        The floating-point number to represent in YAML.

    Returns
    -------
    yaml.ScalarNode
        A YAML scalar node containing the formatted float as a string.

    Notes
    -----
        Precision hardcoded as ``{value:.6g}``
    """
    if value == 0.0:
        return dumper.represent_scalar("tag:yaml.org,2002:int", "0")

    elif value.is_integer():
        text = f"{int(value)}"
        return dumper.represent_scalar("tag:yaml.org,2002:int", text)

    else:
        text = f"{value:.6g}"
        return dumper.represent_scalar("tag:yaml.org,2002:float", text)


def inline_list_representer(dumper, data):
    """
    Get custom YAML representer for writing lists in inline/compact format.

    This function is used as a custom representer for PyYAML to force
    sequences (lists) to be written in inline/compact format.

    Parameters
    ----------
    dumper : yaml.Dumper
        PyYAML Dumper instance handling the serialization.
    data : list
        Python list to be serialized to YAML.

    Returns
    -------
    yaml.nodes.SequenceNode
        YAML sequence node configured for inline representation.
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
