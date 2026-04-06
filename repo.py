#!/usr/bin/env python3

"""Tiny helper to manage this repository."""

# ruff: noqa: D103

import logging
import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

_logger = logging.getLogger(__name__)
_repo_path = Path(__file__).parent


def _paths2shell(paths: Sequence[Path]) -> str:
    # Hopefully no one is that crazy to use colon in the path...
    if any(":" in str(p) for p in paths):
        msg = "Cannot handle colon in paths for extra_paths argument for `run_shell`!"
        raise ValueError(msg)
    return ":".join([str(p) for p in paths])


def shell_command(
    cmd: Sequence[str | Path],
    *,
    extra_env: Mapping[str, str | Path] | None = None,
    extra_paths: Sequence[Path] | None = None,
    capture_output: bool = False,
    cwd: Path | None = None,
) -> str:
    """Return bash equivalent of pure subprocess.run command.

    `subprocess.run` spawns command directly using syscall avoiding any
    shells like bash.
    Sometimes users want to run such commands manually, which requires
    manual parsing of python lists, i.e. `["my", "command", "args"]`
    and then converting them to bash commands.

    This helper function simplifies process and returns str cmd,
    which can be used to supply to bash.

    :param cmd: command to convert.
    :param extra_env: extra environment variables to define.
    :param extra_paths: extra paths to append to PATH variable.
    :param capture_output: whether `subprocess.run` will capture output.
    :param cwd: cwd for spawned command.
    """
    extra_env = extra_env or {}
    extra_paths = extra_paths or []

    if "PATH" in extra_env:
        msg = "Do not pass PATH to extra_env. Use extra_paths instead."
        raise ValueError(msg)

    extra_paths_str = _paths2shell(extra_paths)
    print_cmd = ""

    if cwd is not None and cwd != Path.cwd():
        cwd_str = str(os.path.relpath(cwd, Path.cwd()))  # Use os.path.relpath since it supports "../"
        cwd_str = shlex.quote(cwd_str)
        print_cmd += f"cd {cwd_str} && "

    for k, val in extra_env.items():
        print_cmd += f"{shlex.quote(str(k))}={shlex.quote(str(val))} "

    if extra_paths:
        print_cmd += f'PATH="{extra_paths_str}:${{PATH}}" '

    print_cmd += " ".join([shlex.quote(str(arg)) for arg in cmd])

    if capture_output:
        print_cmd += " &> CAPTURED"

    return print_cmd.strip()


def run_shell(  # noqa: PLR0913
    cmd: Sequence[str | Path],
    *,
    extra_env: Mapping[str, str | Path] | None = None,
    extra_paths: Sequence[Path] | None = None,
    capture_output: bool = False,
    cwd: Path | None = None,
    check: bool = True,
    loglevel: int = logging.INFO,
) -> subprocess.CompletedProcess[str]:
    """Execute `subprocess.run` with different defaults.

    This is a plain wrapper over `subprocess.run`, which
    simplifies common scenario of "just running" a command
    in a bash-like style.

    This util:
    1. Allows to define "extra" environment, instead of replacing it.
    2. Allows to define "extra" PATH, instead of replacing it.
    3. check=True by default.
    4. text=True by default.
    5. Logs supplied command in bash-like form, allowing
       users to run it manually if needed.

    :param cmd: command to run.
    :param extra_env: extra environment variables to define.
    :param extra_paths: extra paths to append to PATH variable.
    :param capture_output: whether to capture output.
    :param cwd: cwd for spawned command.
    :param check: whether to check command exit code.
    :param loglevel: loglevel for dumping bash equivalent of the command.
    """
    extra_env = dict(extra_env) if extra_env is not None else {}
    extra_paths = list(extra_paths) if extra_paths is not None else []

    env = os.environ.copy()
    env.update({k: str(v) for k, v in extra_env.items()})

    extra_paths_str = _paths2shell(extra_paths).strip()
    if extra_paths_str:
        if "PATH" in env and env["PATH"].strip():
            extra_paths_str += ":" + env["PATH"]
        env["PATH"] = extra_paths_str

    print_cmd = shell_command(
        cmd,
        extra_env=extra_env,
        extra_paths=extra_paths,
        capture_output=capture_output,
        cwd=cwd,
    )
    _logger.log(loglevel, f"[RUNNING IN SHELL]: {print_cmd}")
    return subprocess.run(  # noqa: S603
        cmd,
        env=env,
        check=check,
        capture_output=capture_output,
        text=True,
        cwd=cwd,
    )


app = typer.Typer(
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Simple script to manage this repository.",
)


@app.callback()
def setup_app() -> None:
    logger = logging.getLogger()

    handler = RichHandler(show_time=False, show_path=False, show_level=False, markup=True)
    handler.console.stderr = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def git_files(repo_path: Path, *ext: str) -> list[str]:
    return run_shell(
        ["git", "ls-files", *[f"*{e}" for e in ext]], capture_output=True, cwd=repo_path
    ).stdout.splitlines()


def _check_leaked_credentials(repo_path: Path) -> None:
    # In order to properly check for leaked credentials,
    # we have to iterate over all commits in the repo.
    # Pinning first commit ensures, that if the check is run on some shallow cloned repo,
    # it will throw an error.
    first_commit = "1712e58cb568cc877c1115ff57e82ed05ee97d66"

    if run_shell(["git", "cat-file", "-e", first_commit], check=False).returncode:
        _logger.error("Looks like git history is shallow and credential check cannot be performed.")
        raise RuntimeError

    run_shell(["gitleaks", "git"], cwd=repo_path)


@app.command()
def lint() -> None:
    """Lint code."""
    run_shell(["ruff", "check"], cwd=_repo_path)
    # run_shell(["mypy", _repo_path]) # noqa: ERA001

    run_shell(["yamllint", "--strict", _repo_path / ".github"])
    run_shell(["typos"], cwd=_repo_path)

    if not os.getenv("IN_NIX_SHELL"):
        _logger.warning("Not running in nix shell, skipping credential and some other lint checks.")
        return

    _check_leaked_credentials(_repo_path)

    if sh_files := git_files(_repo_path, ".sh"):
        run_shell(["shellcheck", *sh_files], cwd=_repo_path)

    run_shell(["markdownlint-cli2", "."], cwd=_repo_path)
    run_shell(["statix", "check", _repo_path])


@app.command(name="format")
def format_code(
    *,
    check: Annotated[
        bool,
        typer.Option("-c", "--check", help="Only check if code is formatted."),
    ] = False,
) -> None:
    """Format codebase."""
    check_arg = ["--check"] if check else []
    diff_arg = ["--diff"] if check else []
    dry_run_arg = ["--dry-run"] if check else []
    write_arg = ["--write"] if not check else []
    run_shell(["ruff", "format", *check_arg], cwd=_repo_path)
    run_shell(["ruff", "check", "--fix", "--unsafe-fixes", *diff_arg], cwd=_repo_path)

    run_shell(["mdformat", *git_files(_repo_path, ".md"), *check_arg], cwd=_repo_path)

    if not os.getenv("IN_NIX_SHELL"):
        _logger.warning("Not running in nix shell, skipping some format tools.")
        return

    statix_res = run_shell(["statix", "fix", *dry_run_arg, _repo_path], cwd=_repo_path, capture_output=check)
    if check and statix_res.stdout.strip():
        raise RuntimeError(statix_res.stdout)

    run_shell(["nixfmt", "--verify", "--strict", *check_arg, *git_files(_repo_path, ".nix")], cwd=_repo_path)

    run_shell(["shfmt", *write_arg, *diff_arg, _repo_path])
    run_shell(["prettier", *write_arg, _repo_path, *check_arg], cwd=_repo_path)
    run_shell(["stylua", _repo_path, *check_arg], cwd=_repo_path)


if __name__ == "__main__":
    app()
