"""Partition-driven, leakage-safe hyperparameter optimization engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from saber.core.metrics import get_metric_spec, resolve_positive_class, validate_metric
from saber.core.registry import get_algorithm
from saber.exceptions import (
    DatasetValidationError,
    NonFiniteScoreError,
    OptimizationError,
    ValidationContractError,
)
from saber.preprocessing.pipeline import (
    PreprocessingConfig,
    build_model_pipeline,
    pipeline_input,
    preprocessing_summary,
)
from saber.tuning.results import OptimizationResult
from saber.validation.partitioning import (
    EvaluationRole,
    build_explicit_cv,
    resolve_partition_plan,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from saber.core.search_space import SearchSpace
    from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan

OptimizerName = Literal[
    "grid",
    "random",
    "halving_grid",
    "halving_random",
    "optuna",
]


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Optimizer-specific tuning controls.

    Workflow knobs shared with ``validate``/``benchmark`` (``metrics``,
    ``random_state``, ``positive_class``, ...) are keyword arguments of
    :func:`tune`.  ``refit_metric`` selects among those metrics and defaults
    to the first one.
    """

    optimizer: OptimizerName = "grid"
    refit_metric: str | None = None
    refit: bool = True
    n_jobs: int = -1
    n_iter: int = 20
    n_trials: int = 50
    timeout: float | None = None
    factor: int = 3
    resource: str = "n_samples"
    max_resources: str | int = "auto"
    min_resources: str | int = "exhaust"
    aggressive_elimination: bool = False
    error_score: float = np.nan
    optuna_storage: str | None = None
    optuna_study_name: str | None = None
    optuna_load_if_exists: bool = True


def tune(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    model_params: Mapping[str, Any] | None = None,
    preprocessing: PreprocessingConfig | None = None,
    partition_plan: PartitionPlan | BioSievePartitionConfig | None = None,
    config: TuningConfig,
    search_space: SearchSpace | None = None,
    metrics: Sequence[str],
    evaluation_role: EvaluationRole = "auto",
    require_complete: bool = True,
    positive_class: Any | None = None,
    random_state: int | None = None,
) -> OptimizationResult:
    """Tune an algorithm over explicit/BioSieve partitions and return the best result.

    Preprocessing is embedded in the searched sklearn Pipeline so imputation and
    scaling are fitted independently inside each fold. saber never generates
    fallback splitters here: unpartitioned data are delegated to BioSieve.

    Binary positive-class metrics (precision, recall, F1, ROC AUC) are scored
    for ``positive_class``, defaulting to the last sorted class as evaluation does.
    """
    spec = get_algorithm(algorithm)
    dataset.validate(task=spec.task)

    plan = resolve_partition_plan(dataset=dataset, partition_plan=partition_plan)
    plan.validate_against(dataset, require_complete=require_complete)

    search_dataset, cv = build_explicit_cv(
        dataset=dataset,
        plan=plan,
        evaluation_role=evaluation_role,
        require_complete=require_complete,
    )

    search_dataset_ids = search_dataset.resolved_sample_ids
    for split_index, (train_index, _) in enumerate(cv):
        train_ids = tuple(search_dataset_ids[index] for index in train_index)
        try:
            search_dataset.subset(train_ids).validate(task=spec.task)
        except DatasetValidationError as exc:
            raise ValidationContractError(
                f"Training membership for tuning split {split_index} is invalid for task '{spec.task}': {exc}"
            ) from exc

    space = search_space if search_space is not None else spec.search_space
    if space is None:
        raise ValidationContractError(f"No search space defined for algorithm '{spec.name}'.")
    if len(space) == 0:
        raise ValidationContractError(f"Search space for algorithm '{spec.name}' is empty.")

    metrics = tuple(dict.fromkeys(metrics))
    if not metrics:
        raise ValidationContractError("At least one tuning metric is required.")
    refit_metric = config.refit_metric or metrics[0]
    if refit_metric not in metrics:
        raise ValidationContractError(
            f"refit_metric '{refit_metric}' must be included in metrics {metrics!r}."
        )
    scoring_positive_class = resolve_positive_class(
        task=spec.task, y=search_dataset.y, positive_class=positive_class
    )
    scorers = {
        metric: validate_metric(metric, task=spec.task, y=search_dataset.y).make_scorer(
            positive_class=scoring_positive_class
        )
        for metric in metrics
    }

    first_train_index = cv[0][0]
    first_train = search_dataset.subset(tuple(search_dataset_ids[index] for index in first_train_index))
    estimator = spec.build_estimator(
        random_state=random_state,
        **(model_params or {}),
    )
    pipeline = build_model_pipeline(
        spec=spec,
        estimator=estimator,
        training_data=first_train,
        preprocessing=preprocessing,
    )

    fit_params = spec.sample_weight_fit_params(search_dataset.sample_weight)

    runner = _run_optuna if config.optimizer == "optuna" else _run_sklearn
    result = runner(
        spec=spec,
        pipeline=pipeline,
        dataset=search_dataset,
        cv=cv,
        search_space=space,
        scorers=scorers,
        refit_metric=refit_metric,
        config=config,
        random_state=random_state,
        fit_params=fit_params,
    )

    result.partition_plan = plan
    result.metrics = metrics
    result.refit_metric = refit_metric
    result.feature_schema = dataset.feature_schema
    result.positive_class = positive_class
    result.metadata.update(
        {
            "random_state": random_state,
            "preprocessing": preprocessing_summary(preprocessing),
            "tuning_config": asdict(config),
            "dataset_fingerprint": dataset.fingerprint,
            "search_dataset_fingerprint": search_dataset.fingerprint,
            "partition_fingerprint": plan.fingerprint,
            "search_space": space.to_dict(),
        }
    )
    return result


def _run_sklearn(
    *,
    spec: Any,
    pipeline: Any,
    dataset: DatasetBundle,
    cv: tuple[tuple[np.ndarray, np.ndarray], ...],
    search_space: SearchSpace,
    scorers: dict[str, Any],
    refit_metric: str,
    config: TuningConfig,
    random_state: int | None,
    fit_params: dict[str, Any],
) -> OptimizationResult:
    optimizer = config.optimizer
    common: dict[str, Any] = {
        "estimator": pipeline,
        "scoring": scorers,
        "cv": cv,
        "n_jobs": config.n_jobs,
        "refit": refit_metric if config.refit else False,
        "return_train_score": True,
        "error_score": config.error_score,
    }

    if optimizer == "grid":
        search = GridSearchCV(param_grid=_space_params(search_space.to_grid, optimizer), **common)
    elif optimizer == "random":
        search = RandomizedSearchCV(
            param_distributions=_space_params(search_space.to_random, optimizer),
            n_iter=config.n_iter,
            random_state=random_state,
            **common,
        )
    elif optimizer in {"halving_grid", "halving_random"}:
        # sklearn's successive-halving estimators currently accept only a
        # single scoring objective. We optimize the explicit refit metric,
        # then evaluate the selected configuration with all requested
        # metrics on the same explicit folds.
        # Deferred: enables the experimental halving search API only when it is used.
        from sklearn.experimental import enable_halving_search_cv  # noqa: F401, PLC0415

        # sklearn stubs omit these experimental estimators; real once enabled above.
        from sklearn.model_selection import (  # noqa: PLC0415
            HalvingGridSearchCV,  # pyrefly: ignore[missing-module-attribute]
            HalvingRandomSearchCV,  # pyrefly: ignore[missing-module-attribute]
        )

        halving_common = {
            **common,
            "scoring": scorers[refit_metric],
            "refit": config.refit,
            "factor": config.factor,
            "resource": config.resource,
            "max_resources": config.max_resources,
            "aggressive_elimination": config.aggressive_elimination,
        }
        if optimizer == "halving_grid":
            search = HalvingGridSearchCV(
                param_grid=_space_params(search_space.to_grid, optimizer),
                min_resources=config.min_resources,
                **halving_common,
            )
        else:
            search = HalvingRandomSearchCV(
                param_distributions=_space_params(search_space.to_random, optimizer),
                random_state=random_state,
                # HalvingRandomSearchCV's canonical automatic lower-resource policy is "smallest".
                min_resources="smallest" if config.min_resources == "exhaust" else config.min_resources,
                **halving_common,
            )
    else:
        raise ValidationContractError(f"Unknown tuning optimizer '{optimizer}'.")

    try:
        search.fit(pipeline_input(pipeline, dataset.X), dataset.y, **fit_params)
    except ValueError as exc:
        message = str(exc)
        if "fits failed" in message and "All the" in message:
            raise OptimizationError(
                f"Optimizer '{optimizer}' could not fit any candidate for "
                f"algorithm '{spec.name}'. All candidate fits failed."
            ) from exc
        raise
    results = search.cv_results_

    if optimizer in {"grid", "random"}:
        mean_key = f"mean_test_{refit_metric}"
        best_index = _best_index(results[mean_key], spec.name, refit_metric, optimizer)
        best_score = float(results[mean_key][best_index])
        best_scores = {metric: float(results[f"mean_test_{metric}"][best_index]) for metric in scorers}
        history = _history_from_results(results, {metric: metric for metric in scorers})
    else:
        best_index = _best_index(
            results["mean_test_score"],
            spec.name,
            refit_metric,
            optimizer,
        )
        best_score = float(results["mean_test_score"][best_index])
        history = _history_from_results(results, {refit_metric: "score"})
        # sklearn *SearchCV stubs omit `.estimator`, though it's always set
        # from the constructor arg at runtime.
        best_scores = _evaluate_selected_metrics(
            pipeline=search.estimator,  # pyrefly: ignore[missing-attribute]
            params=results["params"][best_index],
            dataset=dataset,
            cv=cv,
            scorers=scorers,
            fit_params=fit_params,
            n_jobs=config.n_jobs,
        )
        best_scores[refit_metric] = best_score

    _validate_best_scores(
        best_scores,
        algorithm=spec.name,
        optimizer=optimizer,
    )
    best_params = _strip_estimator_prefix(dict(results["params"][best_index]))
    best_model = search.best_estimator_ if config.refit else None

    return OptimizationResult(
        algorithm=spec.name,
        optimizer=optimizer,
        refit_metric=refit_metric,
        best_params=best_params,
        best_model=best_model,
        spec=spec,
        history=history,
        study=search,
        best_scores=_natural_scores(best_scores),
    )


def _run_optuna(
    *,
    spec: Any,
    pipeline: Any,
    dataset: DatasetBundle,
    cv: tuple[tuple[np.ndarray, np.ndarray], ...],
    search_space: SearchSpace,
    scorers: dict[str, Any],
    refit_metric: str,
    config: TuningConfig,
    random_state: int | None,
    fit_params: dict[str, Any],
) -> OptimizationResult:
    try:
        import optuna  # noqa: PLC0415  # optuna is an optional dependency
        from optuna.trial import TrialState  # noqa: PLC0415
    except ImportError as exc:
        # Deferred: only needed on the optuna-missing path.
        from saber.exceptions import OptionalDependencyError  # noqa: PLC0415

        raise OptionalDependencyError(
            dependency="optuna",
            extra="optuna",
            purpose="Optuna hyperparameter optimization",
        ) from exc

    sampler = None
    sampler_seed = random_state
    if (
        sampler_seed is not None
        and config.optuna_storage is not None
        and config.optuna_study_name is not None
        and config.optuna_load_if_exists
    ):
        try:
            existing = optuna.load_study(
                study_name=config.optuna_study_name,
                storage=config.optuna_storage,
            )
            # Optuna does not persist sampler RNG state. Advance the seed
            # deterministically on resume so an existing study does not
            # replay the exact same initial suggestions.
            sampler_seed = sampler_seed + len(existing.trials)
        except KeyError:
            pass
    if sampler_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)

    study = optuna.create_study(
        study_name=config.optuna_study_name,
        storage=config.optuna_storage,
        load_if_exists=config.optuna_load_if_exists,
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: Any) -> float:
        params = search_space.sample_optuna(trial)
        pipeline_params = {f"estimator__{name}": value for name, value in params.items()}
        candidate = pipeline.set_params(**pipeline_params)
        try:
            scores = cross_validate(
                candidate,
                pipeline_input(pipeline, dataset.X),
                dataset.y,
                scoring=scorers[refit_metric],
                cv=cv,
                n_jobs=config.n_jobs,
                # sklearn stubs predate the `params=` kwarg (sklearn 1.4+);
                # installed sklearn (1.9.1) supports it.
                params=fit_params or None,  # pyrefly: ignore[unexpected-keyword]
                error_score=config.error_score,
            )["test_score"]
            score = float(np.mean(np.asarray(scores, dtype=float)))
            if not np.isfinite(score):
                raise NonFiniteScoreError(  # noqa: TRY301  # except below records this on the trial
                    algorithm=spec.name,
                    metric=refit_metric,
                    optimizer="optuna",
                    score=score,
                )
        except Exception as exc:
            trial.set_user_attr("saber_error", f"{type(exc).__name__}: {exc}")
            raise
        else:
            return score

    study.optimize(
        objective,
        n_trials=config.n_trials,
        timeout=config.timeout,
        catch=(Exception,),
    )

    completed = [
        (float(trial.value), trial)
        for trial in study.trials
        if trial.state == TrialState.COMPLETE and trial.value is not None and np.isfinite(trial.value)
    ]
    if not completed:
        raise NonFiniteScoreError(
            algorithm=spec.name,
            metric=refit_metric,
            optimizer="optuna",
            score=float("nan"),
        )

    best_score, best_trial = max(completed, key=lambda item: item[0])
    best_params = dict(best_trial.params)
    pipeline_params = {f"estimator__{name}": value for name, value in best_params.items()}
    selected = pipeline.set_params(**pipeline_params)

    best_scores = _evaluate_selected_metrics(
        pipeline=pipeline,
        params=pipeline_params,
        dataset=dataset,
        cv=cv,
        scorers=scorers,
        fit_params=fit_params,
        n_jobs=config.n_jobs,
    )
    best_scores[refit_metric] = best_score
    _validate_best_scores(
        best_scores,
        algorithm=spec.name,
        optimizer="optuna",
    )

    best_model = None
    if config.refit:
        best_model = selected
        best_model.fit(pipeline_input(best_model, dataset.X), dataset.y, **fit_params)

    history: list[dict[str, Any]] = []
    for trial in study.trials:
        score = None if trial.value is None else _natural(refit_metric, trial.value)
        history.append(
            {
                "trial": trial.number,
                "params": dict(trial.params),
                "score": score,
                "metrics": {refit_metric: score},
                "status": {"FAIL": "failed"}.get(trial.state.name, trial.state.name.lower()),
                "error": trial.user_attrs.get("saber_error"),
            }
        )

    return OptimizationResult(
        algorithm=spec.name,
        optimizer="optuna",
        refit_metric=refit_metric,
        best_params=best_params,
        best_model=best_model,
        spec=spec,
        history=history,
        study=study,
        best_scores=_natural_scores(best_scores),
    )


def _space_params(build: Callable[..., Any], optimizer: str) -> Any:
    try:
        return build(prefix="estimator__")
    except ValueError as exc:
        raise ValidationContractError(
            f"Search space for optimizer '{optimizer}' is incompatible: {exc}"
        ) from exc


def _best_index(
    scores: Any,
    algorithm: str,
    metric: str,
    optimizer: str,
) -> int:
    values = np.asarray(scores, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise NonFiniteScoreError(
            algorithm=algorithm,
            metric=metric,
            optimizer=optimizer,
            score=float("nan"),
        )
    candidates = np.where(finite)[0]
    return int(candidates[np.argmax(values[finite])])


def _validate_best_scores(
    scores: Mapping[str, float],
    *,
    algorithm: str,
    optimizer: str,
) -> None:
    for metric, score in scores.items():
        if not np.isfinite(float(score)):
            raise NonFiniteScoreError(
                algorithm=algorithm,
                metric=metric,
                optimizer=optimizer,
                score=float(score),
            )


def _natural(metric: str, score: float) -> float:
    """Convert an internal maximize-oriented score (sklearn/optuna) to the metric's natural sign."""
    return get_metric_spec(metric).to_natural_score(score)


def _natural_scores(scores: Mapping[str, float]) -> dict[str, float]:
    return {metric: _natural(metric, score) for metric, score in scores.items()}


def _strip_estimator_prefix(params: dict[str, Any]) -> dict[str, Any]:
    prefix = "estimator__"
    return {key.removeprefix(prefix): value for key, value in params.items()}


def _history_from_results(
    results: Mapping[str, Any],
    metrics: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Build candidate history; ``metrics`` maps metric name to its cv_results_ key suffix."""
    history: list[dict[str, Any]] = []
    for index, raw_params in enumerate(results["params"]):
        metric_payload = {
            metric: {
                "mean": _natural(metric, results[f"mean_test_{suffix}"][index]),
                "std": float(results[f"std_test_{suffix}"][index]),
                "rank": int(results[f"rank_test_{suffix}"][index]),
            }
            for metric, suffix in metrics.items()
        }
        finite = all(np.isfinite(payload["mean"]) for payload in metric_payload.values())
        history.append(
            {
                "params": _strip_estimator_prefix(dict(raw_params)),
                "metrics": metric_payload,
                "status": "complete" if finite else "failed",
                "error": None if finite else "non-finite candidate score",
                **({"iteration": int(results["iter"][index])} if "iter" in results else {}),
            }
        )
    return history


def _evaluate_selected_metrics(
    *,
    pipeline: Any,
    params: Mapping[str, Any],
    dataset: DatasetBundle,
    cv: tuple[tuple[np.ndarray, np.ndarray], ...],
    scorers: Mapping[str, Any],
    fit_params: Mapping[str, Any],
    n_jobs: int,
) -> dict[str, float]:
    selected = pipeline.set_params(**dict(params))
    scores = cross_validate(
        selected,
        pipeline_input(selected, dataset.X),
        dataset.y,
        scoring=dict(scorers),
        cv=cv,
        n_jobs=n_jobs,
        # sklearn stubs predate the `params=` kwarg (sklearn 1.4+); installed
        # sklearn (1.9.1) supports it.
        params=dict(fit_params) or None,  # pyrefly: ignore[unexpected-keyword]
        error_score=np.nan,
    )
    output: dict[str, float] = {}
    for metric in scorers:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        output[metric] = float(np.mean(values))
    return output
