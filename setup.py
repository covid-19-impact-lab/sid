from pathlib import Path

from setuptools import find_packages
from setuptools import setup


DESCRIPTION = "Simulate the spread of covid-19 with different policies."
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/covid-19-impact-lab/sid/issues",
    "Documentation": "https://sid-.readthedocs.io/en/latest",
    "Source Code": "https://github.com/covid-19-impact-lab/sid",
}
README = Path("README.rst").read_text()


setup(
    name="sid",
    version="0.0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION + "\n\n" + README,
    long_description_content_type="text/x-rst",
    license="None",
    url="https://github.com/covid-19-impact-lab/sid",
    author="Janos Gabler and Tobias Raabe",
    author_email="janos.gabler@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    platforms="any",
    package_data={"sid": ["params.csv", "tests/test_states.csv"]},
    include_package_data=True,
    zip_safe=False,
)
