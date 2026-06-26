from pathlib import Path


def test_dashboard_does_not_use_streamlit_stretch_width() -> None:
    dashboard_files = Path("src/dashboard").rglob("*.py")
    offenders = [
        str(path)
        for path in dashboard_files
        if 'width="stretch"' in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
