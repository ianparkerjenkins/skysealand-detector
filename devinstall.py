"""
A utility script to perform a developer install of the `skysealand` project.
"""

import pathlib
import subprocess
import sys

path = pathlib.Path(__file__).parent

# Dependencies handled via conda environment.
pip_args = ["install", "--no-dependencies", "--no-build-isolation", "--no-index", "-e", str(path)]

if __name__ == "__main__":
    # Set up pre-commit for this project.
    subprocess.call(["pre-commit", "install"])
    # TODO: Add conda stuff here instead?
    sys.exit(subprocess.call(["python", "-m", "pip", *pip_args]))
