"""Verification contract for preserving established ``plot_hdi`` validation outcomes."""

import numpy as np
import pytest
import xarray as xr

import arviz as az
from arviz.plots import hdiplot as hdiplot_module


BACKENDS = ("matplotlib", "bokeh")
DIMENSION_MISMATCH_ERROR = (
    "Dimension mismatch for x: (3,) and hdi: (4, 2). Check the dimensions of y and"
    "hdi_kwargs to make sure they are compatible"
)
INTERVAL_SOURCES = ("computed-from-y", "supplied-hdi-data")
MISSING_SOURCE_ERROR = "One of {y, hdi_data} is required"
MULTIVARIABLE_DATASET_ERROR = (
    "Found several variables in hdi_data. Only single variable Datasets are supported."
)
PROBABILITY_ERROR = "The value of hdi_prob should be in the interval (0, 1]"
SIMULTANEOUS_SOURCE_WARNING = "Both y and hdi_data arguments present, ignoring y"


def _numeric_samples(coordinate_count=3):
    """Return deterministic posterior samples with the requested coordinate dimension."""
    return np.arange(8 * coordinate_count, dtype=float).reshape(2, 4, coordinate_count)


def _precomputed_bounds(coordinate_count=3):
    """Return distinguishable lower and upper interval bounds."""
    lower = np.arange(coordinate_count, dtype=float) * 10
    return np.column_stack((lower, lower + 3))


def _plotting_surface(backend):
    """Create a backend-native surface whose identity establishes the return contract."""
    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        return ax

    from bokeh.plotting import figure

    return figure()


def _renderer_count(surface, backend):
    """Count backend-owned render artifacts on a plotting surface."""
    if backend == "matplotlib":
        return len(surface.lines) + len(surface.collections) + len(surface.patches)
    return len(surface.renderers)


def test_hdicat_014_supported_x_without_y_or_hdi_data_raises_established_missing_interval_source_value_error_before_dispatch(
    monkeypatch,
):
    """GUID: HDICAT-014; one interval source remains mandatory."""

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("missing-source validation must precede HDI computation")

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("missing-source validation must precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)

    with pytest.raises(ValueError) as err:
        az.plot_hdi(np.arange(3, dtype=float), backend="matplotlib", show=False)

    assert str(err.value) == MISSING_SOURCE_ERROR


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_015_supported_x_with_y_and_hdi_data_warns_y_is_ignored_uses_supplied_intervals_and_returns_selected_backend_surface(
    monkeypatch, backend
):
    """GUID: HDICAT-015; supplied intervals remain authoritative when both sources are present."""
    x = np.arange(3, dtype=float)
    supplied_bounds = _precomputed_bounds()
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_selector = hdiplot_module.get_plotting_function
    render_calls = []

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("y must be ignored when hdi_data is supplied")

    def record_selected_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)
        renderer = real_selector(plot_name, module_name, selected_backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return renderer(**kwargs)

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", record_selected_renderer)

    with pytest.warns(UserWarning) as warning_records:
        result = az.plot_hdi(
            x,
            y=_numeric_samples(),
            hdi_data=supplied_bounds,
            smooth=False,
            backend=backend,
            ax=surface,
            show=False,
        )

    assert [str(record.message) for record in warning_records] == [SIMULTANEOUS_SOURCE_WARNING]
    assert len(render_calls) == 1
    np.testing.assert_array_equal(render_calls[0]["x_data"], x)
    np.testing.assert_array_equal(render_calls[0]["y_data"], supplied_bounds)
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)


@pytest.mark.parametrize(
    "hdi_prob", (0, -0.1, 1.1), ids=("zero", "negative", "greater-than-one")
)
def test_hdicat_016_supported_x_with_y_and_hdi_prob_outside_open_zero_closed_one_raises_established_probability_value_error_before_computation_or_dispatch(
    monkeypatch, hdi_prob
):
    """GUID: HDICAT-016; computed intervals retain the established probability domain."""

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("invalid hdi_prob validation must precede HDI computation")

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("invalid hdi_prob validation must precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)

    with pytest.raises(ValueError) as err:
        az.plot_hdi(
            np.arange(3, dtype=float),
            y=_numeric_samples(),
            hdi_prob=hdi_prob,
            backend="matplotlib",
            show=False,
        )

    assert str(err.value) == PROBABILITY_ERROR


def test_hdicat_017_supported_x_with_multivariable_hdi_dataset_raises_established_single_variable_value_error_before_dispatch(
    monkeypatch,
):
    """GUID: HDICAT-017; supplied Dataset intervals remain limited to one data variable."""
    bounds = _precomputed_bounds()
    hdi_data = xr.Dataset(
        {
            "first": (("coordinate", "bound"), bounds),
            "second": (("coordinate", "bound"), bounds + 1),
        }
    )

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("supplied Dataset validation must not compute replacement intervals")

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("Dataset cardinality validation must precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)

    with pytest.raises(ValueError) as err:
        az.plot_hdi(
            np.arange(3, dtype=float),
            hdi_data=hdi_data,
            backend="matplotlib",
            show=False,
        )

    assert str(err.value) == MULTIVARIABLE_DATASET_ERROR


@pytest.mark.parametrize("interval_source", INTERVAL_SOURCES, ids=INTERVAL_SOURCES)
def test_hdicat_018_supported_x_shape_mismatching_computed_or_supplied_hdi_non_bound_dimensions_raises_established_type_error_before_dispatch(
    monkeypatch, interval_source
):
    """GUID: HDICAT-018; both interval sources retain the shared coordinate-shape contract."""
    mismatched_bounds = _precomputed_bounds(coordinate_count=4)
    hdi_calls = []

    def compute_mismatched_hdi(values, **kwargs):
        hdi_calls.append((values, kwargs))
        return mismatched_bounds

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("dimension validation must precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "hdi", compute_mismatched_hdi)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)
    interval_kwargs = (
        {"y": _numeric_samples(coordinate_count=4)}
        if interval_source == "computed-from-y"
        else {"hdi_data": mismatched_bounds}
    )

    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            np.arange(3, dtype=float),
            backend="matplotlib",
            show=False,
            **interval_kwargs,
        )

    assert str(err.value) == DIMENSION_MISMATCH_ERROR
    assert len(hdi_calls) == (1 if interval_source == "computed-from-y" else 0)
