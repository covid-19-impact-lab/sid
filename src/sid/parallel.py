import functools
from pathlib import Path

import numpy as np
from joblib import delayed
from joblib import Parallel
from sid.msm import get_msm_func
from sid.shared import parse_n_workers
from sid.simulate import get_simulate_func


ROWMEAN = functools.partial(np.mean, axis=0)


def get_parallel_msm_func(
    initial_states,
    initial_infections,
    contact_models,
    calc_moments,
    empirical_moments,
    replace_nans,
    n_workers,
    n_evaluations,
    seed,
    duration=None,
    events=None,
    contact_policies=None,
    saved_columns=None,
    initial_conditions=None,
    weighting_matrix=None,
    optional_state_columns=None,
    testing_demand_models=None,
    testing_allocation_models=None,
    testing_processing_models=None,
    value_aggregation_func=np.mean,
    root_contributions_aggregations_func=ROWMEAN,
    path=None,
):
    n_workers = parse_n_workers(n_workers)
    seeds = _draw_seeds(seed, n_evaluations)
    paths = _create_output_directories(path, n_evaluations, seeds)

    return functools.partial(
        _msm_parallel,
        initial_states=initial_states,
        initial_infections=initial_infections,
        contact_models=contact_models,
        duration=duration,
        events=events,
        contact_policies=contact_policies,
        testing_demand_models=testing_demand_models,
        testing_allocation_models=testing_allocation_models,
        testing_processing_models=testing_processing_models,
        seeds=seeds,
        paths=paths,
        saved_columns=saved_columns,
        optional_state_columns=optional_state_columns,
        initial_conditions=initial_conditions,
        calc_moments=calc_moments,
        empirical_moments=empirical_moments,
        replace_nans=replace_nans,
        weighting_matrix=weighting_matrix,
        n_workers=n_workers,
        value_aggregation_func=value_aggregation_func,
        root_contributions_aggregations_func=root_contributions_aggregations_func,
    )


def _msm_parallel(
    params,
    initial_states,
    initial_infections,
    contact_models,
    duration,
    events,
    contact_policies,
    testing_demand_models,
    testing_allocation_models,
    testing_processing_models,
    seeds,
    paths,
    saved_columns,
    optional_state_columns,
    initial_conditions,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix,
    n_workers,
    value_aggregation_func,
    root_contributions_aggregations_func,
):
    outs = Parallel(n_jobs=n_workers)(
        delayed(_msm_worker)(
            params,
            initial_states,
            initial_infections,
            contact_models,
            duration,
            events,
            contact_policies,
            testing_demand_models,
            testing_allocation_models,
            testing_processing_models,
            seed,
            path,
            saved_columns,
            optional_state_columns,
            initial_conditions,
            calc_moments,
            empirical_moments,
            replace_nans,
            weighting_matrix,
        )
        for seed, path in zip(seeds, paths)
    )

    result = {
        "value": value_aggregation_func([out["value"] for out in outs]),
        "root_contributions": root_contributions_aggregations_func(
            [out["root_contributions"] for out in outs]
        ),
        "empirical_moments": [out["empirical_moments"] for out in outs],
        "simulated_moments": [out["simulated_moments"] for out in outs],
    }

    return result


def _msm_worker(
    params,
    initial_states,
    initial_infections,
    contact_models,
    duration,
    events,
    contact_policies,
    testing_demand_models,
    testing_allocation_models,
    testing_processing_models,
    seed,
    path,
    saved_columns,
    optional_state_columns,
    initial_conditions,
    calc_moments,
    empirical_moments,
    replace_nans,
    weighting_matrix,
):
    simulate = get_simulate_func(
        params,
        initial_states,
        initial_infections,
        contact_models,
        duration,
        events,
        contact_policies,
        testing_demand_models,
        testing_allocation_models,
        testing_processing_models,
        seed,
        path,
        saved_columns,
        optional_state_columns,
        initial_conditions,
    )
    msm = get_msm_func(
        simulate,
        calc_moments,
        empirical_moments,
        replace_nans,
        weighting_matrix,
    )

    result = msm(params)

    return result


def _draw_seeds(seed, n_evaluations):
    np.random.seed(seed)
    return np.random.randint(0, 1_000_000, size=n_evaluations)


def _create_output_directories(paths, n_evaluations, seeds):
    if paths is None:
        paths = [Path.cwd() / f".sid-{i}" for i in seeds]
    elif isinstance(paths, list):
        paths = [Path(path) for path in paths]
    else:
        raise ValueError("Either pass a list of paths or pass None.")

    if len(paths) != n_evaluations:
        raise ValueError("There must be as many paths as evaluations.")

    return paths
