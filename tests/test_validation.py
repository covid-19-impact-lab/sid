from contextlib import ExitStack as does_not_raise  # noqa: N813

import pytest
from sid.validation import _validate_model_function
from sid.validation import validate_contact_policies
from sid.validation import validate_testing_models
from sid.validation import validate_vaccination_models


@pytest.mark.integration
@pytest.mark.parametrize(
    "contact_policies, contact_models, expectation",
    [
        pytest.param(
            None,
            {},
            pytest.raises(ValueError, match="'contact_policies' must be "),
            id="test is dictionary.",
        ),
        pytest.param(
            {"model": None},
            {},
            pytest.raises(ValueError, match="Contact policy"),
            id="test model is dictionary.",
        ),
        pytest.param(
            {"model": {}},
            {},
            pytest.raises(ValueError, match="The contact policy"),
            id="test missing keys.",
        ),
        pytest.param(
            {
                "model": {
                    "policy": None,
                    "start": None,
                    "end": None,
                    "affected_contact_model": None,
                }
            },
            {},
            pytest.raises(ValueError, match="The contact policy 'model' affects"),
            id="test affected model is present",
        ),
        pytest.param(
            {
                "model": {
                    "policy": lambda x: None,
                    "start": None,
                    "end": None,
                    "affected_contact_model": "contact_model",
                }
            },
            {"contact_model": None},
            pytest.raises(ValueError, match="The model 'model' of 'contact_policies'"),
            id="test function",
        ),
        pytest.param(
            {
                "model": {
                    "policy": 0.5,
                    "start": None,
                    "end": None,
                    "affected_contact_model": "model",
                }
            },
            {"model": {"is_recurrent": True}},
            pytest.raises(ValueError, match="Specifying multipliers"),
            id="test number for recurrent models.",
        ),
        pytest.param(
            {
                "model": {
                    "policy": -1,
                    "start": None,
                    "end": None,
                    "affected_contact_model": "model",
                }
            },
            {"model": {"is_recurrent": False}},
            pytest.raises(ValueError, match="The policy of contact policy 'model'"),
            id="test number for random models.",
        ),
        pytest.param(
            {
                "model": {
                    "policy": None,
                    "start": None,
                    "end": None,
                    "affected_contact_model": "model",
                }
            },
            {"model": {"is_recurrent": False}},
            pytest.raises(ValueError, match="The 'policy' entry of contact policy"),
            id="test invalid policy.",
        ),
    ],
)
def test_validate_contact_policies(contact_policies, contact_models, expectation):
    with expectation:
        validate_contact_policies(contact_policies, contact_models)


@pytest.mark.integration
@pytest.mark.parametrize(
    "demand, allocation, processing, expectation",
    [
        (
            None,
            {},
            {},
            pytest.raises(ValueError, match="'testing_demand_models' must be"),
        ),
        ({"model": None}, {}, {}, pytest.raises(ValueError, match="Each model")),
        ({"model": {}}, {}, {}, pytest.raises(TypeError, match="The model")),
    ],
)
def test_validate_testing_models(demand, allocation, processing, expectation):
    with expectation:
        validate_testing_models(demand, allocation, processing)


@pytest.mark.integration
@pytest.mark.parametrize(
    "models, expectation",
    [
        (None, pytest.raises(ValueError, match="'vaccination_models' must be")),
        ({"model": None}, pytest.raises(ValueError, match="Vaccination model")),
        (
            {"model": {"model": lambda receives_vaccine, states, params, seed: None}},
            does_not_raise(),
        ),
    ],
)
def test_validate_vaccination_models(models, expectation):
    with expectation:
        validate_vaccination_models(models)


@pytest.mark.unit
@pytest.mark.parametrize(
    "model_name, model_group, model, args, expectation",
    [
        pytest.param(
            "a",
            "group_a",
            lambda x: True,
            ["x"],
            does_not_raise(),
            id="Pass and all arguments matched.",
        ),
        pytest.param(
            "b",
            "group_b",
            lambda x, y: True,
            ["x"],
            does_not_raise(),
            id="Pass only subset of params matched.",
        ),
        pytest.param(
            "c",
            "group_c",
            None,
            None,
            pytest.raises(TypeError, match="The model 'c' of 'group_c' is not a "),
            id="error when model is not a callable.",
        ),
        pytest.param(
            "d",
            "group_d",
            lambda x: True,
            ["x", "y"],
            pytest.raises(ValueError, match="The model 'd' of 'group_d' is missing"),
            id="error when not all arguments are accepted by the function.",
        ),
    ],
)
def test__validate_model_function(model_name, model_group, model, args, expectation):
    with expectation:
        _validate_model_function(model_name, model_group, model, args)
