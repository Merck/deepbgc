from deepbgc.main import run
import pytest


def test_unit_main_help():
    with pytest.raises(SystemExit) as excinfo:
        run(['--help'])
    assert excinfo.value.code == 0


def test_unit_pipeline_help():
    with pytest.raises(SystemExit) as excinfo:
        run(['pipeline', '--help'])
    assert excinfo.value.code == 0


def test_unit_prepare_help():
    with pytest.raises(SystemExit) as excinfo:
        run(['prepare', '--help'])
    assert excinfo.value.code == 0


def test_unit_train_help():
    with pytest.raises(SystemExit) as excinfo:
        run(['train', '--help'])
    assert excinfo.value.code == 0


def test_unit_main_invalid_command():
    with pytest.raises(SystemExit) as excinfo:
        run(['invalid'])
    assert excinfo.value.code == 2