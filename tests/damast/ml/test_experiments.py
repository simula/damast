import pytest

from damast.ml.experiments import Experiment


@pytest.fixture()
def experiment_dir(tmp_path):
    return tmp_path


def test_validate_experiment_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Experiment.validate_experiment_dir(dir="UNKNOWN_DIR")

    with pytest.raises(NotADirectoryError):
        filename = tmp_path / "dummy-file"
        with open(filename, "w") as f:
            pass

        Experiment.validate_experiment_dir(dir=filename)

    with pytest.raises(NotADirectoryError, match="is not an experiment"):
        Experiment.validate_experiment_dir(dir=tmp_path)

    Experiment.touch_marker(dir=tmp_path)
    Experiment.validate_experiment_dir(dir=tmp_path)
