from pathlib import Path

from setuptools import find_packages
from setuptools import setup


DESCRIPTION = "Simulate the spread of COVID-19 with different policies."
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/covid-19-impact-lab/sid/issues",
    "Documentation": "https://sid-dev.readthedocs.io/en/latest",
    "Source Code": "https://github.com/covid-19-impact-lab/sid",
}
README = Path("README.rst").read_text()


setup(
    name="sid",
    version="0.0.1",
    description=DESCRIPTION,
    long_description=DESCRIPTION + "\n\n" + README,
    long_description_content_type="text/x-rst",
    license="MIT",
    url="https://github.com/covid-19-impact-lab/sid",
    author="Gabler, Raabe, Röhrl",
    author_email="janos.gabler@gmail.com",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    platforms="any",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"sid": ["covid_epi_params.csv"]},
    include_package_data=True,
    zip_safe=False,
)
