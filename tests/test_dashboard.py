import importlib.util
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1] / "dashboard"


def _load_data_access():
    sys.path.insert(0, str(DASH))
    spec = importlib.util.spec_from_file_location("data_access", DASH / "data_access.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loaders_handle_missing_files(tmp_path):
    da = _load_data_access()
    assert da.load_json(tmp_path / "nope.json") is None
    assert da.load_table(tmp_path / "nope.parquet") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert da.load_json(tmp_path / "bad.json") is None


def test_artifact_status_lists_all_stages():
    da = _load_data_access()
    s = da.artifact_status()
    assert len(s) == len(da.ARTIFACTS) and s.available.dtype == bool


def test_dashboard_app_compiles():
    compile((DASH / "app.py").read_text(encoding="utf-8"), "app.py", "exec")


def test_dashboard_runs_headless_on_artifacts():
    """Executes the whole app script (all tabs) against the generated artifacts, if present."""
    import pytest

    da = _load_data_access()
    if not da.artifact_status().available.all():
        pytest.skip("pipeline artifacts not generated yet")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASH / "app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 7
