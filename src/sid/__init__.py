import logging

from sid.colors import get_colors
from sid.initial_conditions import sample_initial_immunity
from sid.initial_conditions import sample_initial_infections
from sid.msm import get_msm_func
from sid.shared import get_epidemiological_parameters
from sid.shared import load_epidemiological_parameters
from sid.simulate import get_simulate_func
from sid.time import get_date
from sid.time import sid_period_to_timestamp
from sid.time import timestamp_to_sid_period

from ._version import get_versions


__version__ = get_versions()["version"]
del get_versions


__all__ = [
    "sample_initial_infections",
    "sample_initial_immunity",
    "get_colors",
    "get_date",
    "get_epidemiological_parameters",
    "get_msm_func",
    "get_simulate_func",
    "load_epidemiological_parameters",
    "sid_period_to_timestamp",
    "timestamp_to_sid_period",
]


logger = logging.getLogger("sid")

if not logging.root.handlers:
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
