"""Compatibility helpers for supported Streamlit width APIs."""

from __future__ import annotations

from functools import wraps
from inspect import Parameter, signature
from typing import Any, Callable


def _accepts_stretch_width(function: Callable[..., Any]) -> bool:
    """Return whether a Streamlit command natively accepts ``width="stretch"``."""
    try:
        width = signature(function).parameters.get("width")
    except (TypeError, ValueError):
        return False
    return bool(
        width is not None
        and width.kind not in {Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL}
        and width.default == "stretch"
    )


def install_streamlit_width_compat(streamlit_module: Any) -> None:
    """Translate the new stretch-width API when running Streamlit 1.36–1.49.

    Streamlit 1.50+ accepts ``width="stretch"`` for dataframes and Plotly
    charts. Earlier supported releases require ``use_container_width=True``;
    passing the string to ``st.dataframe`` is interpreted as an integer and
    raises ``TypeError``. The adapter is idempotent and leaves modern releases
    untouched.
    """
    if getattr(streamlit_module, "_northern_signal_width_compat", False):
        return

    original_dataframe = streamlit_module.dataframe
    if not _accepts_stretch_width(original_dataframe):

        @wraps(original_dataframe)
        def dataframe_compat(*args: Any, width: Any = None, **kwargs: Any) -> Any:
            if width == "stretch":
                kwargs.setdefault("use_container_width", True)
            elif width == "content":
                kwargs.setdefault("use_container_width", False)
            elif width is not None:
                kwargs["width"] = width
            return original_dataframe(*args, **kwargs)

        streamlit_module.dataframe = dataframe_compat

    original_plotly_chart = streamlit_module.plotly_chart
    if not _accepts_stretch_width(original_plotly_chart):

        @wraps(original_plotly_chart)
        def plotly_chart_compat(*args: Any, width: Any = None, **kwargs: Any) -> Any:
            if width == "stretch":
                kwargs.setdefault("use_container_width", True)
            elif width == "content":
                kwargs.setdefault("use_container_width", False)
            return original_plotly_chart(*args, **kwargs)

        streamlit_module.plotly_chart = plotly_chart_compat

    streamlit_module._northern_signal_width_compat = True
