from setuptools import setup, find_packages

setup(
    name="consul-poisoning-ml",
    version="1.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        # add other dependencies here
    ],
)