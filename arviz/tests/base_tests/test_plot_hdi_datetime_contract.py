"""Verification contract for preserving NumPy datetime HDI behavior."""

import numpy as np
import pytest

import arviz as az
from arviz.plots import hdiplot as hdiplot_module


BACKENDS = ("matplotlib", "bokeh")
DATETIME_ERROR = "Cannot deal with x as type datetime. Recommend setting smooth=False."
DATETIME_PLACEHOLDER = pytest.mark.skip(
    reason="HDICAT-012/013 datetime HDI placeholders: activate in Malkhut"
)
INTERVAL_SOURCES = ("computed-from-y", "supplied-hdi-data")


def _unsorted_datetime_x():
    """Return NumPy datetime coordinates whose order makes paired sorting observable."""
    return np.asarray(["2022-03-01", "2022-01-01", "2022-02-01"], dtype="datetime64[D]")


def _interval_kwargs(interval_source):
    """Return shape-compatible posterior samples or distinguishable supplied bounds."""
    if interval_source == "computed-from-y":
        return {"y": np.arange(24, dtype=float).reshape(2, 4, 3)}
    return {"hdi_data": np.asarray([[30.0, 33.0], [10.0, 13.0], [20.0, 23.0]])}


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


@DATETIME_PLACEHOLDER
@pytest.mark.parametrize("interval_source", INTERVAL_SOURCES, ids=INTERVAL_SOURCES)
def test_hdicat_012_smoothed_numpy_datetime_x_with_computed_or_precomputed_intervals_raises_datetime_guidance_before_backend_dispatch(
    monkeypatch, interval_source
):
    """GUID: HDICAT-012; smoothing datetime x retains its dedicated public guidance."""

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("smoothed datetime validation must precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)

    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            _unsorted_datetime_x(),
            smooth=True,
            backend="matplotlib",
            show=False,
            **_interval_kwargs(interval_source),
        )

    assert str(err.value) == DATETIME_ERROR
    assert "categorical" not in str(err.value).lower()
    assert "string" not in str(err.value).lower()


@DATETIME_PLACEHOLDER
@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_013_unsmoothed_unsorted_numpy_datetime_x_and_precomputed_intervals_sort_together_render_and_return_selected_backend_surface(
    monkeypatch, backend
):
    """GUID: HDICAT-013; unsmoothed datetime pairs render on and return both backends."""
    x = _unsorted_datetime_x()
    supplied_bounds = _interval_kwargs("supplied-hdi-data")["hdi_data"]
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_selector = hdiplot_module.get_plotting_function
    render_calls = []

    def record_selected_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)
        renderer = real_selector(plot_name, module_name, selected_backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return renderer(**kwargs)

        return render

    monkeypatch.setattr(hdiplot_module, "get_plotting_function", record_selected_renderer)

    result = az.plot_hdi(
        x,
        hdi_data=supplied_bounds,
        smooth=False,
        backend=backend,
        ax=surface,
        show=False,
    )

    expected_x = np.asarray(["2022-01-01", "2022-02-01", "2022-03-01"], dtype="datetime64[D]")
    expected_bounds = np.asarray([[10.0, 13.0], [20.0, 23.0], [30.0, 33.0]])
    assert len(render_calls) == 1
    np.testing.assert_array_equal(render_calls[0]["x_data"], expected_x)
    np.testing.assert_array_equal(render_calls[0]["y_data"], expected_bounds)
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)
