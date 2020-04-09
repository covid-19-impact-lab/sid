from setuptools import find_packages
from setuptools import setup


setup(
    name="simulator",
    version="0.0.1",
    description=("Simulate the spread of covid-19 with different policies."),
    license="MIT",
    url="https://github.com/covid-19-impact-lab/simulator",
    author="Janos Gabler and Tobias Raabe",
    author_email="janos.gabler@gmail.com",
    packages=find_packages(),
    zip_safe=False,
)
