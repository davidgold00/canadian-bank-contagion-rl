from __future__ import annotations

from src.dashboard.streamlit_compat import install_streamlit_width_compat


class LegacyStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def dataframe(
        self,
        data=None,
        width: int | None = None,
        *,
        use_container_width: bool | None = None,
        **kwargs,
    ):
        self.calls.append(("dataframe", width, use_container_width))
        return data

    def plotly_chart(self, figure, use_container_width: bool = True, **kwargs):
        self.calls.append(("plotly_chart", None, use_container_width))
        return figure


class ModernStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def dataframe(
        self,
        data=None,
        width="stretch",
        *,
        use_container_width: bool | None = None,
        **kwargs,
    ):
        self.calls.append(("dataframe", width, use_container_width))
        return data

    def plotly_chart(
        self,
        figure,
        *,
        width="stretch",
        use_container_width: bool | None = None,
        **kwargs,
    ):
        self.calls.append(("plotly_chart", width, use_container_width))
        return figure


def test_legacy_streamlit_translates_stretch_width() -> None:
    streamlit = LegacyStreamlit()

    install_streamlit_width_compat(streamlit)
    streamlit.dataframe([], width="stretch")
    streamlit.plotly_chart("figure", width="stretch")

    assert streamlit.calls == [
        ("dataframe", None, True),
        ("plotly_chart", None, True),
    ]


def test_modern_streamlit_keeps_native_width_api() -> None:
    streamlit = ModernStreamlit()

    install_streamlit_width_compat(streamlit)
    streamlit.dataframe([], width="stretch")
    streamlit.plotly_chart("figure", width="stretch")

    assert streamlit.calls == [
        ("dataframe", "stretch", None),
        ("plotly_chart", "stretch", None),
    ]


def test_install_is_idempotent() -> None:
    streamlit = LegacyStreamlit()

    install_streamlit_width_compat(streamlit)
    first_dataframe = streamlit.dataframe
    first_plotly_chart = streamlit.plotly_chart
    install_streamlit_width_compat(streamlit)

    assert streamlit.dataframe is first_dataframe
    assert streamlit.plotly_chart is first_plotly_chart
