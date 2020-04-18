from setuptools import find_packages
from setuptools import setup


setup(
    name="sid",
    version="0.0.1",
    description="Simulate the spread of covid-19 with different policies.",
    license="None",
    url="https://github.com/covid-19-impact-lab/sid",
    author="Janos Gabler and Tobias Raabe",
    author_email="janos.gabler@gmail.com",
    packages=find_packages(),
    zip_safe=False,
)
