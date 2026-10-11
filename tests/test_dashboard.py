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
    """Executes the whole app script (all tabs) against the generated artifacts, if present.

    Runs in a subprocess so that a native crash of the Streamlit runtime is reported as a test
    failure instead of terminating the whole pytest session."""
    import subprocess

    import pytest

    da = _load_data_access()
    if not da.artifact_status().available.all():
        pytest.skip("pipeline artifacts not generated yet")
    code = (
        "from streamlit.testing.v1 import AppTest\n"
        f"at = AppTest.from_file({str(DASH / 'app.py')!r}, default_timeout=180).run()\n"
        "def ok():\n"
        "    assert not at.exception, [e.value for e in at.exception]\n"
        "ok()\n"
        "assert [t.label for t in at.tabs] == ['Decision Support', 'Store Overview', 'Results', "
        "'About the Project', 'Technical Details'], [t.label for t in at.tabs]\n"
        # non-default choices: rule recommender, one product in the overview, actual-sales results,
        # every planning week, and every cost what-if chart
        "at.radio[0].set_value('Rule: cover a high estimate').run(); ok()\n"
        "at.selectbox(key='ov_prod').select_index(1).run(); ok()\n"
        "at.radio[1].set_value('Actual past sales').run(); ok()\n"
        "for i in range(len(at.selectbox[0].options)):\n"
        "    at.selectbox[0].select_index(i).run(); ok()\n"
        "for i in range(len(at.selectbox[-1].options)):\n"
        "    at.selectbox[-1].select_index(i).run(); ok()\n"
        "print('APPTEST_OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "APPTEST_OK" in r.stdout, (
        f"exit code {r.returncode} (3221225477 / 139 = native access violation)\n{r.stderr[-2000:]}")


def _load_explain():
    sys.path.insert(0, str(DASH))
    spec = importlib.util.spec_from_file_location("explain", DASH / "explain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_risk_status_covers_all_risk_classes():
    ex = _load_explain()
    got = {c: ex.risk_status(c) for c in ["high stockout risk", "below upper bound", "excess", "balanced",
                                          "unknown stock"]}
    assert got == {"high stockout risk": "Stockout risk", "below upper bound": "Stockout risk",
                   "excess": "Excess stock", "balanced": "Normal", "unknown stock": "No stock record"}
    assert set(got.values()) <= set(ex.RISK_COLORS)


def test_transfer_reason_mentions_stores_quantity_and_cost():
    ex = _load_explain()
    r = ex.transfer_reason("Store 7", "Store 12", 15, 42.3, 27.04, 30, 4.2, 1, 3.5, 0.86)
    assert "Store 12 is likely to run out" in r and "Store 7 has spare stock" in r
    assert "Move 15 units over 42 km" in r and "27.04" in r
    assert "may run short" in ex.transfer_reason("A", "B", 1, 5, 9, 3, 1, 2, 1.2, 0.3)
    assert "no stock record" in ex.transfer_reason("A", "B", 1, 5, 9, 3, 1, 0, 1.2, 1.0, dst_no_record=True)


def test_best_policy_prefers_no_transfers_when_all_lose_money():
    ex = _load_explain()
    assert ex.best_policy({"a": -5.0, "b": -1.0}) == "No transfers"
    assert ex.best_policy({"a": -5.0, "b": 3.0}) == "b"


def test_no_code_terms_in_user_facing_labels():
    ex = _load_explain()
    text = " ".join(list(ex.POLICY_NAMES.values()) + list(ex.COST_PARAM_NAMES.values()) + list(ex.RISK_HELP.values()))
    for term in ("seed", "checkpoint", "parquet", "json"):
        assert term not in text.lower()


def test_out_of_stock_wording():
    ex = _load_explain()
    assert "Store 9 is out of stock" in ex.transfer_reason("Store 1", "Store 9", 1, 10, 5, 4, 0.4, 0, 0.4, 1.0)
    assert "Move 1 unit over" in ex.transfer_reason("Store 1", "Store 9", 1, 10, 5, 4, 0.4, 0, 0.4, 1.0)
    assert "is out of stock" in ex.no_transfer_reason("Store 9", "Stockout risk", 1.0, 0, 0.4)
