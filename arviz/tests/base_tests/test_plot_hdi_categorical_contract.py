"""Verification contract for rejecting categorical strings before HDI rendering."""

import numpy as np
import pytest
from pandas import Categorical

import arviz as az
from arviz.plots import hdiplot as hdiplot_module


pytestmark = pytest.mark.skip(
    reason="HDICAT contract placeholder: activate during implementation validation"
)

HDICAT_ERROR = "Categorical or string x values are unsupported."
X_REPRESENTATIONS = ("numpy-string", "pandas-categorical-object")
INTERVAL_SOURCES = ("computed-from-y", "supplied-hdi-data")
SMOOTH_MODES = (True, False)


def _categorical_x(representation):
    """Return each categorical-string representation named by the contract."""
    labels = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
    if representation == "numpy-string":
        return np.asarray(labels, dtype="U5")
    return np.asarray(Categorical(labels), dtype=object)


def _interval_kwargs(interval_source):
    """Return shape-compatible computed or precomputed interval input."""
    if interval_source == "computed-from-y":
        return {"y": np.arange(64, dtype=float).reshape(2, 4, 8)}
    return {"hdi_data": np.column_stack((np.arange(8), np.arange(8) + 1))}


@pytest.mark.parametrize("representation", X_REPRESENTATIONS, ids=X_REPRESENTATIONS)
def test_hdicat_001_categorical_string_representations_raise_intentional_type_error(
    representation,
):
    """GUID: HDICAT-001; categorical-string x raises the documented public error."""
    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            _categorical_x(representation),
            smooth=False,
            **_interval_kwargs("computed-from-y"),
        )

    assert str(err.value) == HDICAT_ERROR


@pytest.mark.parametrize("representation", X_REPRESENTATIONS, ids=X_REPRESENTATIONS)
@pytest.mark.parametrize("smooth", SMOOTH_MODES, ids=("smooth", "unsmoothed"))
def test_hdicat_002_computed_intervals_reject_categorical_string_x_for_each_smoothing_mode(
    representation, smooth
):
    """GUID: HDICAT-002; computed intervals reject strings with or without smoothing."""
    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            _categorical_x(representation),
            smooth=smooth,
            **_interval_kwargs("computed-from-y"),
        )

    assert str(err.value) == HDICAT_ERROR


@pytest.mark.parametrize("representation", X_REPRESENTATIONS, ids=X_REPRESENTATIONS)
@pytest.mark.parametrize("smooth", SMOOTH_MODES, ids=("smooth", "unsmoothed"))
def test_hdicat_003_supplied_intervals_reject_categorical_string_x_for_each_smoothing_mode(
    representation, smooth
):
    """GUID: HDICAT-003; supplied intervals reject strings with or without smoothing."""
    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            _categorical_x(representation),
            smooth=smooth,
            **_interval_kwargs("supplied-hdi-data"),
        )

    assert str(err.value) == HDICAT_ERROR


@pytest.mark.parametrize("representation", X_REPRESENTATIONS, ids=X_REPRESENTATIONS)
@pytest.mark.parametrize("interval_source", INTERVAL_SOURCES, ids=INTERVAL_SOURCES)
@pytest.mark.parametrize("smooth", SMOOTH_MODES, ids=("smooth", "unsmoothed"))
@pytest.mark.parametrize("namespace", ("arviz", "arviz.plots"))
@pytest.mark.parametrize(
    "backend,configured",
    (
        ("matplotlib", False),
        ("bokeh", False),
        ("matplotlib", True),
        ("bokeh", True),
    ),
    ids=(
        "explicit-matplotlib",
        "explicit-bokeh",
        "configured-matplotlib",
        "configured-bokeh",
    ),
)
def test_hdicat_004_public_namespaces_and_backend_selections_reject_before_dispatch(
    monkeypatch, representation, interval_source, smooth, namespace, backend, configured
):
    """GUID: HDICAT-004; shared frontend validation precedes every backend dispatch."""

    def fail_if_dispatched(*args, **kwargs):
        pytest.fail("categorical-string validation did not precede backend dispatch")

    monkeypatch.setattr(hdiplot_module, "get_plotting_function", fail_if_dispatched)
    plot_hdi = az.plot_hdi if namespace == "arviz" else az.plots.plot_hdi
    backend_arg = None if configured else backend

    with az.rc_context({"plot.backend": backend}):
        with pytest.raises(TypeError) as err:
            plot_hdi(
                _categorical_x(representation),
                smooth=smooth,
                backend=backend_arg,
                **_interval_kwargs(interval_source),
            )

    assert str(err.value) == HDICAT_ERROR


def _plotting_surface(backend):
    """Create a caller-owned plotting surface with existing render state."""
    if backend == "matplotlib":
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        ax.fill_between([0, 1], [0, 0], [1, 1])
        ax.add_patch(Rectangle((0, 0), 0.25, 0.25))
        return ax

    from bokeh.plotting import figure

    ax = figure()
    ax.line([0, 1], [0, 1])
    ax.patch([0, 1, 0], [0, 1, 1])
    return ax


def _surface_state(ax, backend):
    """Capture identity-preserving renderer collections for later comparison."""
    if backend == "matplotlib":
        return (tuple(ax.lines), tuple(ax.collections), tuple(ax.patches))
    return tuple(ax.renderers)


@pytest.mark.parametrize("representation", X_REPRESENTATIONS, ids=X_REPRESENTATIONS)
@pytest.mark.parametrize("interval_source", INTERVAL_SOURCES, ids=INTERVAL_SOURCES)
@pytest.mark.parametrize("smooth", SMOOTH_MODES, ids=("smooth", "unsmoothed"))
@pytest.mark.parametrize("backend", ("matplotlib", "bokeh"))
def test_hdicat_005_caller_surface_state_is_unchanged_when_categorical_string_x_is_rejected(
    representation, interval_source, smooth, backend
):
    """GUID: HDICAT-005; rejection leaves a caller-owned plotting surface untouched."""
    ax = _plotting_surface(backend)
    state_before = _surface_state(ax, backend)

    with pytest.raises(TypeError) as err:
        az.plot_hdi(
            _categorical_x(representation),
            smooth=smooth,
            backend=backend,
            ax=ax,
            show=False,
            **_interval_kwargs(interval_source),
        )

    assert str(err.value) == HDICAT_ERROR
    assert _surface_state(ax, backend) == state_before

    if backend == "matplotlib":
        import matplotlib.pyplot as plt

        plt.close(ax.figure)
