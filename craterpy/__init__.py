"""craterpy module"""
from pathlib import Path


def get_version():
    """Return version from pyproject.toml"""
    pyproj_path = Path(__file__).parent.parent.joinpath("pyproject.toml")
    with Path(pyproj_path).open("r") as f:
        line = f.readline()
        while "version" not in line.lower():
            line = f.readline()
        if line:
            return line.split("=")[1].strip()
        return "?.?.?"


__version__ = get_version()
