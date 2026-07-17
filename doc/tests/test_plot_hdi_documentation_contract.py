"""Tests for the user-facing ``plot_hdi`` documentation contract."""

import ast
from pathlib import Path


def _plot_hdi_x_description():
    """Return the ``x`` parameter description without importing optional plotting dependencies."""
    source = Path(__file__).parents[2] / "arviz" / "plots" / "hdiplot.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    plot_hdi = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "plot_hdi"
    )
    docstring = ast.get_docstring(plot_hdi)
    lines = docstring.splitlines()
    try:
        x_entry = lines.index("x : array-like")
    except ValueError as err:
        raise AssertionError("plot_hdi docstring must document the x parameter") from err

    description = []
    for line in lines[x_entry + 1 :]:
        if line and not line.startswith(" "):
            break
        if line.strip():
            description.append(line.strip())
    return " ".join(description).lower()


# USER-FACING DOCUMENTATION CONTRACT [HDI-005]: verify the public x-parameter entry separately
# from the runtime categorical-input contract.
def test_hdi_005_when_accepted_x_inputs_are_described_categorical_x_is_explicitly_unsupported():
    """Document categorical x as unsupported in accepted plot_hdi inputs [HDI-005]."""
    assert "categorical ``x`` inputs are unsupported" in _plot_hdi_x_description()


def test_hdi_005_when_accepted_x_inputs_are_described_string_valued_x_is_explicitly_unsupported():
    """Document string-valued x as unsupported when plot_hdi inputs are described [HDI-005]."""
    assert "string-valued ``x`` inputs are also unsupported" in _plot_hdi_x_description()


def test_hdi_005_given_the_documented_restriction_categorical_axis_support_is_not_implied():
    """Avoid claiming or implying categorical-axis plotting support [HDI-005]."""
    description = _plot_hdi_x_description()

    assert description == (
        "numeric axis values to plot. categorical ``x`` inputs are unsupported. string-valued "
        "``x`` inputs are also unsupported."
    )


def test_hdi_005_documented_accepted_x_is_consistent_with_numeric_axis_values():
    """Keep the documented plot_hdi x restriction consistent with numeric axis values [HDI-005]."""
    assert _plot_hdi_x_description().startswith("numeric axis values to plot.")
