from deepbgc.main import main
import pytest


def test_unit_main_help():
    with pytest.raises(SystemExit) as excinfo:
        main(['--help'])
    assert excinfo.value.code == 0


def test_unit_pipeline_help():
    with pytest.raises(SystemExit) as excinfo:
        main(['pipeline', '--help'])
    assert excinfo.value.code == 0


def test_unit_prepare_help():
    with pytest.raises(SystemExit) as excinfo:
        main(['prepare', '--help'])
    assert excinfo.value.code == 0


def test_unit_train_help():
    with pytest.raises(SystemExit) as excinfo:
        main(['train', '--help'])
    assert excinfo.value.code == 0


def test_unit_main_invalid_command():
    with pytest.raises(SystemExit) as excinfo:
        main(['invalid'])
    assert excinfo.value.code == 2