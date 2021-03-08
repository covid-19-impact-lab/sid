from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sid.config import DTYPE_VIRUS_STRAIN


def combine_first_factorized_infections(
    first: np.ndarray, second: np.ndarray
) -> np.ndarray:
    """Combine factorized infections where the first has precedence."""
    combined = second.copy()
    combined[first >= 0] = first[first >= 0]
    return combined


def categorize_factorized_infections(
    factorized_infections: Union[pd.Series, np.ndarray], virus_strains: Dict[str, Any]
) -> pd.Series:
    """Convert factorized infections with virus strains to a categorical."""
    return pd.Series(
        pd.Categorical(
            factorized_infections, categories=range(-1, len(virus_strains["names"]))
        )
        .rename_categories(["not_infected"] + virus_strains["names"])
        .remove_categories("not_infected")
    )


def factorize_initial_infections(
    infections: pd.DataFrame, virus_strains: Dict[str, Any]
) -> pd.DataFrame:
    """Factorize multiple boolean or categorical infections."""
    all_columns_boolean = (infections.dtypes == np.bool).all()
    only_one_virus = len(virus_strains["names"]) == 1

    all_columns_categorical = (infections.dtypes == "category").all()

    if (all_columns_boolean and only_one_virus) or all_columns_categorical:
        factorized_infections = pd.DataFrame(index=infections.index)
        for column in infections.columns:
            values = factorize_boolean_or_categorical_infections(
                infections[column], virus_strains
            )
            factorized_infections[column] = values

    else:
        raise ValueError("Infections are not all boolean or categorical.")

    return factorized_infections


def factorize_boolean_or_categorical_infections(infections, virus_strains):
    """Factorize boolean or categorical infections."""
    if pd.core.dtypes.common.is_bool_dtype(infections):
        values, _ = _factorize_boolean_infections(infections, virus_strains["names"])
    elif pd.core.dtypes.common.is_categorical_dtype(infections):
        values, _ = factorize_categorical_infections(infections, virus_strains["names"])
    else:
        raise ValueError(
            "Unknown dtype of infections. Can only handle 'bool' and 'category'"
        )

    return values


def _factorize_boolean_infections(
    infected: Union[pd.Series, np.ndarray], names: List[str]
) -> Tuple[np.ndarray]:
    """Factorize boolean infection."""
    if len(names) > 1:
        raise ValueError(
            f"Boolean infections must correspond to one virus strain, but got {names}."
        )
    if infected.dtype.name != "bool":
        raise ValueError("Infections must have a bool dtype.")

    values = np.full(len(infected), -1, dtype=DTYPE_VIRUS_STRAIN)
    values[infected] = 0
    categories = np.array(names[:1])
    return values, categories


def factorize_categorical_infections(
    virus_strain: pd.Series, names: List[str]
) -> Tuple[np.ndarray]:
    """Factorize a categorical variable indicating virus strains."""
    try:
        virus_strain = virus_strain.cat.reorder_categories(names)
    except ValueError as e:
        raise ValueError(
            "Infections do not align with the passed virus strains:\n\n"
            f"virus_strains: {names}\ninfections: {virus_strain.cat.categories}"
        ) from e
    return (
        virus_strain.cat.codes.to_numpy(DTYPE_VIRUS_STRAIN),
        virus_strain.cat.categories,
    )
