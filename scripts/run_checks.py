#!/usr/bin/env python3
"""Run static analysis, tests, and coverage checks."""

import os
import subprocess
import sys

# =============================
# Configuration
# =============================
ENV_NAME = "qubo-facility-location"
PYLINT_MIN_SCORE = 6.5
COVERAGE_MIN = 65
STEPS = 4


# =============================
# Conda
# =============================
def find_conda() -> str:
    """Locate a working Conda executable.

    Preference order:
      1. CONDA_EXE environment variable.
      2. 'conda' available on PATH.

    Returns:
        Path or command corresponding to a working Conda executable.

    Raises:
        RuntimeError: If no working Conda executable can be found.
    """
    candidates = []

    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(conda_exe)

    candidates.append("conda")

    checked = set()

    for candidate in candidates:
        if candidate in checked:
            continue

        checked.add(candidate)

        try:
            subprocess.run(
                [candidate, "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return candidate

        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    raise RuntimeError(
        "Could not locate a working Conda executable. "
        "Please make sure Conda is installed and accessible."
    )


# =============================
# UI
# =============================
SEP = "-" * 60
BIGSEP = "=" * 60


def supports_ansi() -> bool:
    """Return whether the current output stream supports ANSI formatting."""
    return sys.stdout.isatty()


class C:
    """ANSI color codes used by the command-line interface."""

    if supports_ansi():
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
    else:
        GREEN = RED = YELLOW = BLUE = RESET = ""


def info(msg: str) -> None:
    """Print an informational message."""
    print(f"{C.BLUE}{msg}{C.RESET}")


def ok(msg: str) -> None:
    """Print a successful check message."""
    print(f"{C.GREEN}OK:{C.RESET} {msg}")


def warn(msg: str) -> None:
    """Print a warning message."""
    print(f"{C.YELLOW}WARNING:{C.RESET} {msg}")


def err(msg: str) -> None:
    """Print an error message."""
    print(f"{C.RED}ERROR:{C.RESET} {msg}")


# =============================
# Helpers
# =============================
def run(cmd: list[str]) -> int:
    """Execute a command and return its exit code."""
    info(f"$ {' '.join(cmd)}")
    return subprocess.call(cmd)


def conda_run(conda_exe: str, args: list[str]) -> int:
    """Execute a command inside the configured Conda environment.

    Args:
        conda_exe: Path or command for the Conda executable.
        args: Command and arguments to execute inside the environment.

    Returns:
        Exit code returned by the command.
    """
    return run(
        [
            conda_exe,
            "run",
            "-n",
            ENV_NAME,
            *args,
        ]
    )


def env_exists(conda_exe: str) -> bool:
    """Check whether the configured Conda environment exists.

    Args:
        conda_exe: Path or command for the Conda executable.

    Returns:
        True if the configured environment exists, otherwise False.
    """
    process = subprocess.run(
        [conda_exe, "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )

    if process.returncode != 0:
        return False

    for line in process.stdout.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        name = line.split()[0]

        if name == ENV_NAME:
            return True

    return False


# =============================
# Pipeline
# =============================
def step(title: str, idx: int) -> None:
    """Print the header for a pipeline step."""
    print()
    info(SEP)
    info(f"[{idx}/{STEPS}] {title}")
    info(SEP)


def main() -> int:
    """Run the repository quality checks."""
    info(BIGSEP)
    info(f"  Checks pipeline (environment: {ENV_NAME})")
    info(BIGSEP)

    # Check Conda
    print()
    info(SEP)
    info("Checking Conda...")
    info(SEP)

    try:
        conda_exe = find_conda()
    except RuntimeError as error:
        err(str(error))
        return 1

    ok(f"Using Conda executable: {conda_exe}")

    # Check environment
    print()
    info(SEP)
    info(f'Checking environment "{ENV_NAME}"...')
    info(SEP)

    if not env_exists(conda_exe):
        err(f'Conda environment "{ENV_NAME}" does not exist.')
        warn("Run these commands:")
        print()
        print("  conda env create -f environment.yml")
        print(f"  conda activate {ENV_NAME}")
        print()
        return 1

    ok(f"Environment {ENV_NAME} found")

    fail = 0

    # [1/4] pylint
    step(f"Running pylint (min={PYLINT_MIN_SCORE})...", 1)

    rc = conda_run(
        conda_exe,
        [
            "pylint",
            "src/",
            "executables/",
            "tests/",
            "scripts/",
            f"--fail-under={PYLINT_MIN_SCORE}",
        ],
    )

    if rc != 0:
        err(
            f"pylint failed. "
            f"Threshold {PYLINT_MIN_SCORE} not reached"
        )
        fail = 1
    else:
        ok("pylint passed")

    # [2/4] mypy
    step("Running mypy...", 2)

    rc = conda_run(
        conda_exe,
        [
            "mypy",
            "src/",
            "tests/",
            "executables/",
            "scripts/",
            "--install-types",
            "--non-interactive",
            "--ignore-missing-imports",
        ],
    )

    if rc != 0:
        err("mypy failed")
        fail = 1
    else:
        ok("mypy passed")

    # [3/4] tests
    step("Running tests...", 3)

    rc = conda_run(
        conda_exe,
        [
            "coverage",
            "run",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
        ],
    )

    if rc != 0:
        err("Tests failed")
        fail = 1
    else:
        ok("Tests passed")

    # [4/4] coverage
    step(
        f"Coverage report (min={COVERAGE_MIN}%)...",
        4,
    )

    rc = conda_run(
        conda_exe,
        [
            "coverage",
            "report",
            "-m",
            f"--fail-under={COVERAGE_MIN}",
        ],
    )

    if rc != 0:
        warn(
            f"Insufficient coverage. "
            f"Minimum required {COVERAGE_MIN}%"
        )
        fail = 1
    else:
        ok("Coverage OK")

    # Final summary
    print()
    info(BIGSEP)

    if fail == 0:
        ok(f"Checks finished. Project {ENV_NAME} ready")
        info(BIGSEP)
        return 0

    warn("There were failures in the analysis or tests.")
    info(BIGSEP)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
