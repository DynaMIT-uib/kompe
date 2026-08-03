"""Validate that a release tag agrees with the package version."""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def main(tag: str) -> None:
    """Exit with an error unless ``tag`` is ``v<project.version>``."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    expected = f"v{version}"
    if tag != expected:
        raise SystemExit(f"Release tag {tag!r} does not match package version {expected!r}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_release_tag.py TAG")
    main(sys.argv[1])
