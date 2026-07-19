"""Verification contract for preserving computed and precomputed numeric HDIs."""

import numpy as np
import pytest

import arviz as az
from arviz.plots import hdiplot as hdiplot_module


BACKENDS = ("matplotlib", "bokeh")
CUSTOM_SMOOTH_KWARGS = {"window_length": 21, "polyorder": 3, "mode": "mirror"}


def _numeric_samples(coordinate_count=8):
    """Return deterministic, shape-compatible posterior samples for numeric coordinates."""
    return np.linspace(-1, 1, 8 * coordinate_count).reshape(2, 4, coordinate_count)


def _precomputed_bounds(coordinate_count=8):
    """Return distinguishable, shape-compatible lower and upper interval bounds."""
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


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
@pytest.mark.parametrize(
    "configured_backend", (False, True), ids=("explicit-backend", "configured-backend")
)
def test_hdicat_009_compatible_precomputed_bounds_bypass_hdi_computation_and_return_selected_backend_surface(
    monkeypatch, backend, configured_backend
):
    """GUID: HDICAT-009; supplied numeric bounds remain authoritative through rendering."""
    x = np.arange(8, dtype=float)
    supplied_bounds = _precomputed_bounds()
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_selector = hdiplot_module.get_plotting_function
    render_calls = []

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("shape-compatible precomputed hdi_data must bypass HDI computation")

    def record_selected_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)
        renderer = real_selector(plot_name, module_name, selected_backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return renderer(**kwargs)

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", record_selected_renderer)
    backend_arg = None if configured_backend else backend

    with az.rc_context({"plot.backend": backend}):
        result = az.plot_hdi(
            x,
            hdi_data=supplied_bounds,
            smooth=False,
            backend=backend_arg,
            ax=surface,
            show=False,
        )

    assert len(render_calls) == 1
    np.testing.assert_array_equal(render_calls[0]["x_data"], x)
    np.testing.assert_array_equal(render_calls[0]["y_data"], supplied_bounds)
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)


def test_hdicat_010_default_smoothing_interpolates_precomputed_bounds_on_regular_grid_then_applies_default_savgol_filter(
    monkeypatch,
):
    """GUID: HDICAT-010; supplied bounds retain the established default smoothing sequence."""
    x = np.asarray([0.0, 0.7, 1.3, 2.4, 4.0, 5.1, 6.8, 8.0])
    supplied_bounds = _precomputed_bounds()
    real_griddata = hdiplot_module.griddata
    real_savgol_filter = hdiplot_module.savgol_filter
    grid_calls = []
    filter_calls = []
    render_calls = []
    surface = object()

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("smoothing precomputed hdi_data must not invoke HDI computation")

    def record_griddata(points, values, grid, *args, **kwargs):
        result = real_griddata(points, values, grid, *args, **kwargs)
        grid_calls.append((points, values, grid, args, kwargs, result))
        return result

    def record_savgol_filter(values, *args, **kwargs):
        result = real_savgol_filter(values, *args, **kwargs)
        filter_calls.append((values, args, kwargs, result))
        return result

    def select_renderer(plot_name, module_name, backend):
        assert (plot_name, module_name, backend) == ("plot_hdi", "hdiplot", "matplotlib")

        def render(**kwargs):
            render_calls.append(kwargs)
            return surface

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "griddata", record_griddata)
    monkeypatch.setattr(hdiplot_module, "savgol_filter", record_savgol_filter)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", select_renderer)

    result = az.plot_hdi(
        x, hdi_data=supplied_bounds, smooth=True, backend="matplotlib", show=False
    )

    expected_grid = np.linspace(x.min(), x.max(), 200)
    expected_grid[0] = (expected_grid[0] + expected_grid[1]) / 2
    assert len(grid_calls) == len(filter_calls) == len(render_calls) == 1
    np.testing.assert_array_equal(grid_calls[0][0], x)
    np.testing.assert_array_equal(grid_calls[0][1], supplied_bounds)
    np.testing.assert_array_equal(grid_calls[0][2], expected_grid)
    np.testing.assert_array_equal(filter_calls[0][0], grid_calls[0][5])
    assert filter_calls[0][1] == ()
    assert filter_calls[0][2] == {"axis": 0, "window_length": 55, "polyorder": 2}
    np.testing.assert_array_equal(render_calls[0]["x_data"], expected_grid)
    np.testing.assert_array_equal(render_calls[0]["y_data"], filter_calls[0][3])
    assert result is surface


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_010_caller_smooth_kwargs_filter_precomputed_bounds_without_hdi_recomputation_and_render_selected_backend(
    monkeypatch, backend
):
    """GUID: HDICAT-010; caller filtering options remain authoritative for supplied bounds."""
    x = np.asarray([0.0, 0.7, 1.3, 2.4, 4.0, 5.1, 6.8, 8.0])
    supplied_bounds = _precomputed_bounds()
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_savgol_filter = hdiplot_module.savgol_filter
    real_selector = hdiplot_module.get_plotting_function
    filter_calls = []
    render_calls = []

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("caller-smoothed precomputed hdi_data must not invoke HDI computation")

    def record_savgol_filter(values, *args, **kwargs):
        result = real_savgol_filter(values, *args, **kwargs)
        filter_calls.append((values, args, kwargs, result))
        return result

    def record_selected_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)
        renderer = real_selector(plot_name, module_name, selected_backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return renderer(**kwargs)

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "savgol_filter", record_savgol_filter)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", record_selected_renderer)

    result = az.plot_hdi(
        x,
        hdi_data=supplied_bounds,
        smooth=True,
        smooth_kwargs=dict(CUSTOM_SMOOTH_KWARGS),
        backend=backend,
        ax=surface,
        show=False,
    )

    assert len(filter_calls) == 1
    assert filter_calls[0][1] == ()
    assert filter_calls[0][2] == {"axis": 0, **CUSTOM_SMOOTH_KWARGS}
    assert len(render_calls) == 1
    np.testing.assert_array_equal(render_calls[0]["y_data"], filter_calls[0][3])
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_011_unsmoothed_unsorted_coordinates_and_precomputed_bounds_are_sorted_together_before_rendering(
    monkeypatch, backend
):
    """GUID: HDICAT-011; supplied lower and upper bounds retain their coordinate pairing."""
    x = np.asarray([3.0, 1.0, 2.0])
    supplied_bounds = np.asarray([[30.0, 33.0], [10.0, 13.0], [20.0, 23.0]])
    render_calls = []
    surface = object()

    def fail_if_hdi_computed(*args, **kwargs):
        pytest.fail("sorting precomputed hdi_data must not invoke HDI computation")

    def select_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return surface

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", fail_if_hdi_computed)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", select_renderer)

    result = az.plot_hdi(
        x, hdi_data=supplied_bounds, smooth=False, backend=backend, show=False
    )

    assert len(render_calls) == 1
    np.testing.assert_array_equal(render_calls[0]["x_data"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        render_calls[0]["y_data"], [[10.0, 13.0], [20.0, 23.0], [30.0, 33.0]]
    )
    assert result is surface


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
@pytest.mark.parametrize(
    "configured_backend", (False, True), ids=("explicit-backend", "configured-backend")
)
@pytest.mark.parametrize(
    "hdi_prob,configured_hdi_prob",
    ((0.60, 0.80), (None, 0.70)),
    ids=("explicit-hdi-prob", "configured-hdi-prob"),
)
def test_hdicat_008_numeric_samples_compute_configured_or_explicit_single_hdi_with_options_and_return_selected_backend_surface(
    monkeypatch, backend, configured_backend, hdi_prob, configured_hdi_prob
):
    """GUID: HDICAT-008; numeric samples preserve HDI options and backend return objects."""
    x = np.arange(8, dtype=float)
    samples = _numeric_samples()
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_hdi = hdiplot_module.hdi
    hdi_calls = []

    def record_hdi(values, **kwargs):
        result = real_hdi(values, **kwargs)
        hdi_calls.append((values, kwargs, result))
        return result

    monkeypatch.setattr(hdiplot_module, "hdi", record_hdi)
    backend_arg = None if configured_backend else backend

    with az.rc_context(
        {"plot.backend": backend, "stats.ci_prob": configured_hdi_prob}
    ):
        result = az.plot_hdi(
            x,
            y=samples,
            hdi_prob=hdi_prob,
            circular=True,
            hdi_kwargs={"skipna": True},
            smooth=False,
            backend=backend_arg,
            ax=surface,
            show=False,
        )

    expected_hdi_prob = configured_hdi_prob if hdi_prob is None else hdi_prob
    assert len(hdi_calls) == 1
    np.testing.assert_array_equal(hdi_calls[0][0], samples)
    assert hdi_calls[0][1] == {
        "hdi_prob": expected_hdi_prob,
        "circular": True,
        "multimodal": False,
        "skipna": True,
    }
    assert hdi_calls[0][2].shape == x.shape + (2,)
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)


def test_hdicat_010_default_smoothing_interpolates_regular_grid_then_applies_default_savgol_filter_before_rendering(
    monkeypatch,
):
    """GUID: HDICAT-010; default smoothing preserves the grid and filter sequence."""
    x = np.asarray([0.0, 0.7, 1.3, 2.4, 4.0, 5.1, 6.8, 8.0])
    samples = _numeric_samples()
    real_griddata = hdiplot_module.griddata
    real_savgol_filter = hdiplot_module.savgol_filter
    grid_calls = []
    filter_calls = []
    render_calls = []
    surface = object()

    def record_griddata(points, values, grid, *args, **kwargs):
        result = real_griddata(points, values, grid, *args, **kwargs)
        grid_calls.append((points, values, grid, args, kwargs, result))
        return result

    def record_savgol_filter(values, *args, **kwargs):
        result = real_savgol_filter(values, *args, **kwargs)
        filter_calls.append((values, args, kwargs, result))
        return result

    def select_renderer(plot_name, module_name, backend):
        assert (plot_name, module_name, backend) == ("plot_hdi", "hdiplot", "matplotlib")

        def render(**kwargs):
            render_calls.append(kwargs)
            return surface

        return render

    monkeypatch.setattr(hdiplot_module, "griddata", record_griddata)
    monkeypatch.setattr(hdiplot_module, "savgol_filter", record_savgol_filter)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", select_renderer)

    result = az.plot_hdi(x, y=samples, smooth=True, backend="matplotlib", show=False)

    expected_grid = np.linspace(x.min(), x.max(), 200)
    expected_grid[0] = (expected_grid[0] + expected_grid[1]) / 2
    assert len(grid_calls) == len(filter_calls) == len(render_calls) == 1
    np.testing.assert_array_equal(grid_calls[0][0], x)
    np.testing.assert_array_equal(grid_calls[0][2], expected_grid)
    np.testing.assert_array_equal(filter_calls[0][0], grid_calls[0][5])
    assert filter_calls[0][1] == ()
    assert filter_calls[0][2] == {"axis": 0, "window_length": 55, "polyorder": 2}
    np.testing.assert_array_equal(render_calls[0]["x_data"], expected_grid)
    np.testing.assert_array_equal(render_calls[0]["y_data"], filter_calls[0][3])
    assert result is surface


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_010_caller_smooth_kwargs_are_honored_and_selected_backend_surface_is_returned(
    monkeypatch, backend
):
    """GUID: HDICAT-010; caller filter options reach smoothing and preserve backend returns."""
    x = np.asarray([0.0, 0.7, 1.3, 2.4, 4.0, 5.1, 6.8, 8.0])
    samples = _numeric_samples()
    surface = _plotting_surface(backend)
    renderers_before = _renderer_count(surface, backend)
    real_savgol_filter = hdiplot_module.savgol_filter
    filter_calls = []

    def record_savgol_filter(values, *args, **kwargs):
        filter_calls.append((values, args, kwargs))
        return real_savgol_filter(values, *args, **kwargs)

    monkeypatch.setattr(hdiplot_module, "savgol_filter", record_savgol_filter)

    result = az.plot_hdi(
        x,
        y=samples,
        smooth=True,
        smooth_kwargs=dict(CUSTOM_SMOOTH_KWARGS),
        backend=backend,
        ax=surface,
        show=False,
    )

    assert len(filter_calls) == 1
    assert filter_calls[0][1] == ()
    assert filter_calls[0][2] == {"axis": 0, **CUSTOM_SMOOTH_KWARGS}
    assert _renderer_count(surface, backend) > renderers_before
    assert result is surface

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(surface.figure)


@pytest.mark.parametrize("backend", BACKENDS, ids=BACKENDS)
def test_hdicat_011_unsmoothed_unsorted_coordinates_and_computed_bounds_are_sorted_together_before_rendering(
    monkeypatch, backend
):
    """GUID: HDICAT-011; unsmoothed coordinates retain their computed interval pairing."""
    x = np.asarray([3.0, 1.0, 2.0])
    samples = _numeric_samples(coordinate_count=3)
    computed_hdi = np.asarray([[30.0, 31.0], [10.0, 11.0], [20.0, 21.0]])
    hdi_calls = []
    render_calls = []
    surface = object()

    def compute_hdi(values, **kwargs):
        hdi_calls.append((values, kwargs))
        return computed_hdi

    def select_renderer(plot_name, module_name, selected_backend):
        assert (plot_name, module_name, selected_backend) == ("plot_hdi", "hdiplot", backend)

        def render(**kwargs):
            render_calls.append(kwargs)
            return surface

        return render

    monkeypatch.setattr(hdiplot_module, "hdi", compute_hdi)
    monkeypatch.setattr(hdiplot_module, "get_plotting_function", select_renderer)

    result = az.plot_hdi(x, y=samples, smooth=False, backend=backend, show=False)

    assert len(hdi_calls) == len(render_calls) == 1
    np.testing.assert_array_equal(hdi_calls[0][0], samples)
    np.testing.assert_array_equal(render_calls[0]["x_data"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        render_calls[0]["y_data"], [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    )
    assert result is surface
