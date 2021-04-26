import logging

from sid import statistics
from sid.colors import get_colors
from sid.initial_conditions import sample_initial_immunity
from sid.initial_conditions import sample_initial_infections
from sid.msm import get_msm_func
from sid.shared import load_epidemiological_parameters
from sid.simulate import get_simulate_func
from sid.time import get_date
from sid.time import sid_period_to_timestamp
from sid.time import timestamp_to_sid_period

try:
    from ._version import version as __version__
except ImportError:
    # broken installation, we don't even try unknown only works because we do poor mans
    # version compare
    __version__ = "unknown"


__all__ = [
    "__version__",
    "sample_initial_infections",
    "sample_initial_immunity",
    "get_colors",
    "get_date",
    "get_msm_func",
    "get_simulate_func",
    "load_epidemiological_parameters",
    "sid_period_to_timestamp",
    "timestamp_to_sid_period",
    "statistics",
]


logger = logging.getLogger("sid")

if not logging.root.handlers:
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        handler = logging.StreamHandler()
        logger.addHandler(handler)
