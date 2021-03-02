from typing import Any
from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
from sid.config import DTYPE_VIRUS_STRAIN


def combine_first_factorized_infections(
    first: Union[pd.Series, np.ndarray], second: Union[pd.Series, np.ndarray]
) -> Union[pd.Series, np.ndarray]:
    """Combine factorized infections where the first has precedence."""
    combined = second.copy()
    combined[first >= 0] = first[first >= 0]
    return combined


def categorize_factorized_infections(
    factorized_infections: Union[pd.Series, np.ndarray], virus_strains: Dict[str, Any]
) -> pd.Series:
    return (
        pd.Categorical(
            factorized_infections, categories=range(-1, len(virus_strains["names"]))
        )
        .rename_categories(["not_infected"] + virus_strains["names"])
        .remove_categories("not_infected")
    )


def factorize_multiple_boolean_or_categorical_infections(
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
    if infections.dtype == np.bool:
        values, categories = _factorize_boolean_infections(infections)
    elif infections.dtype == "category":
        values, categories = _factorize_categorical_infections(infections)
    else:
        raise ValueError(
            "Unknown dtype of infections. Can only handle 'bool' and 'category'"
        )

    if not (categories == virus_strains["names"]).all():
        raise ValueError(
            "Infections do not align with the passed virus strains:\n\n"
            f"virus_strains: {virus_strains['names']}\nparsed: {categories}"
        )

    return values


def _factorize_boolean_infections(
    infected: Union[pd.Series, np.ndarray]
) -> Tuple[np.ndarray]:
    values = np.full(len(infected), -1, dtype=DTYPE_VIRUS_STRAIN)
    values[infected] = 0
    categories = np.array(["base_strain"])
    return values, categories


def _factorize_categorical_infections(virus_strain: pd.Series) -> Tuple[np.ndarray]:
    """Factorize a categorical variable indicating virus strains."""
    values, categories = pd.factorize(virus_strain, sort=True)
    values = values.astype(DTYPE_VIRUS_STRAIN)
    return values, categories
