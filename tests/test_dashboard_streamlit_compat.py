from pathlib import Path


def test_dashboard_does_not_use_deprecated_container_width() -> None:
    dashboard_files = Path("src/dashboard").rglob("*.py")
    offenders = [
        str(path)
        for path in dashboard_files
        if path.name != "streamlit_compat.py"
        if "use_container_width" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
