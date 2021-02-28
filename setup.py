from pathlib import Path

from setuptools import find_packages
from setuptools import setup

import versioneer


DESCRIPTION = "Simulate the spread of COVID-19 with different policies."
PROJECT_URLS = {
    "Tracker": "https://github.com/covid-19-impact-lab/sid/issues",
    "Documentation": "https://sid-dev.readthedocs.io/en/latest",
    "Github": "https://github.com/covid-19-impact-lab/sid",
    "Changelog": "https://sid-dev.readthedocs.io/en/latest/changes.html",
}
README = Path("README.rst").read_text()


setup(
    name="sid",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=DESCRIPTION,
    long_description=DESCRIPTION + "\n\n" + README,
    long_description_content_type="text/x-rst",
    license="MIT",
    author="Gabler, Raabe, Röhrl",
    author_email="janos.gabler@gmail.com",
    url=PROJECT_URLS["Github"],
    project_urls=PROJECT_URLS,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    install_requires=[
        "bokeh",
        "dask",
        "fastparquet",
        "numba >= 0.48",
        "numpy",
        "pandas >= 1",
        "python-snappy",
        "seaborn",
        "tqdm",
    ],
    python_requires=">=3.6",
    platforms="any",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    zip_safe=False,
)
