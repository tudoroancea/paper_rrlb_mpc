from setuptools import find_packages, setup

setup(
    name="rrlb",
    packages=find_packages(include=["rrlb", "cstr", "mass_chain"]),
)
