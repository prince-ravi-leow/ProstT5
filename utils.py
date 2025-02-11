"""Misc utils"""

import json
import shlex
import subprocess
import sys
from pathlib import Path

# Test if the current platform is Windows or not
IS_WINDOWS = sys.platform.startswith("win")


def _run_command(command, **kwargs):
    """
    Basic shell command runner
    Credits: Neil Scheidwasser-Clow
    """
    if IS_WINDOWS:
        command = f"wsl {command}"

    subprocess_args = {
        "capture_output": kwargs.get("capture_output", True),
        "shell": kwargs.get("shell", not IS_WINDOWS),
    }

    try:
        output = subprocess.run(command, check=True, **subprocess_args)
    except subprocess.CalledProcessError as _:
        # pylint: disable=subprocess-run-check
        output = subprocess.run(command, **subprocess_args)
        # pylint: enable=subprocess-run-check

        raise RuntimeError(output) from _


def check_input(input_dir, ext):
    """Check files with provided extension exist in provided directory"""
    assert Path(input_dir).is_dir(), "Invalid input dir, please check again"
    assert len(list(Path(input_dir).glob(f"*.{ext}"))) != 0, (
        f"No files with supplied file extension {ext}, please check again"
    )


def parse_json_to_args(config_path):
    """Parse JSON file into a list of command-line arguments."""
    with open(config_path, "rt", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_args = [
        str(key) if value is None else f"{key} {value}"
        for key, value in config_dict.items()
    ]
    return shlex.split(" ".join(config_args))
