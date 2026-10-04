from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")


def test_dashboard_waits_for_solve():
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert len(at.info) == 1


def test_dashboard_solves_with_default_parameters():
    at = AppTest.from_file(APP).run(timeout=60)
    at.button[0].click().run(timeout=120)
    assert not at.exception
    assert not at.error
    assert len(at.dataframe) >= 1


def test_dashboard_reports_pathwise_gamma_error():
    at = AppTest.from_file(APP).run(timeout=60)
    gamma_box = next(sb for sb in at.selectbox if sb.label == "Gamma estimator")
    gamma_box.select("pathwise")
    at.button[0].click().run(timeout=120)
    assert len(at.error) == 1


def test_dashboard_solves_an_american_put():
    at = AppTest.from_file(APP).run(timeout=60)
    next(sb for sb in at.selectbox if sb.label == "Exercise").select("American")
    at.run(timeout=60)
    at.button[0].click().run(timeout=300)
    assert not at.exception
    assert not at.error
    assert len(at.dataframe) >= 1
