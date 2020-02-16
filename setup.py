"""Setup script for omnisolver project."""
from setuptools import setup, find_packages


setup(
    name="omnisolver",
    entry_points={
        "console_scripts": ["omnisolver=omnisolver.cmd:main"]
    },
    install_requires=["dimod"],
    tests_require=["pytest"],
    packages=find_packages(exclude=["tests"]),
)
