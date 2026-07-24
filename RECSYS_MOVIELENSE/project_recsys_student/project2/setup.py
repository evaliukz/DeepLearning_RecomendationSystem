"""Setuptools shim so `pip install -e .` works on older pip (<21.3) too.

The project metadata lives in pyproject.toml; this file just lets legacy
editable installs succeed on systems with an older pip (e.g. the macOS system
Python's pip 21.x).
"""

from setuptools import find_packages, setup

setup(
    name="tinytransformer",
    version="0.2.0",
    description="Project 2 -- sequential recommendation on MovieLens with your Project-1 transformer",
    packages=find_packages(include=["tinytransformer", "tinytransformer.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.23",
        "scikit-learn>=1.1",
        "matplotlib>=3.5",
    ],
)
