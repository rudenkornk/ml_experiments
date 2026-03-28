from ml_experiments.utils import run_shell


def test_cli() -> None:
    run_shell(["python", "-m", "ml_experiments", "pass"])
