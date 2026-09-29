"""Partition-driven, leakage-safe hyperparameter optimization engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate

from saber.core.registry import AlgorithmRegistry
from saber.core.search_space import SearchSpace
from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from saber.exceptions import (
    DatasetValidationError,
    NonFiniteScoreError,
    OptimizationError,
    ValidationContractError,
)
from saber.preprocessing import PreprocessingConfig, build_model_pipeline, pipeline_input
from saber.tuning.results import OptimizationResult
from saber.tuning.scorers import get_scorer
from saber.validation.partitioning import (
    EvaluationRole,
    build_explicit_cv,
    resolve_partition_plan,
)

OptimizerName = Literal[
    "grid",
    "random",
    "halving_grid",
    "halving_random",
    "optuna",
]


@dataclass(frozen=True, slots=True)
class TuningConfig:
    """Backend-neutral tuning controls."""

    optimizer: OptimizerName = "grid"
    metrics: tuple[str, ...] = ()
    refit_metric: str | None = None
    refit: bool = True
    n_jobs: int = -1
    random_state: int | None = None
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

    def resolved_refit_metric(self) -> str:
        if not self.metrics:
            raise ValidationContractError("At least one tuning metric is required.")
        metric = self.refit_metric or self.metrics[0]
        if metric not in self.metrics:
            raise ValidationContractError(
                f"refit_metric '{metric}' must be included in metrics {self.metrics!r}."
            )
        return metric


class TuningEngine:
    """Tune a registered estimator using explicit/BioSieve partitions.

    Preprocessing is embedded in the searched sklearn Pipeline so imputation and
    scaling are fitted independently inside each fold. saber never generates
    fallback splitters here: unpartitioned data are delegated to BioSieve.
    """

    def __init__(self, registry: AlgorithmRegistry) -> None:
        self.registry = registry

    def run(
        self,
        *,
        dataset: DatasetBundle,
        algorithm: str,
        config: TuningConfig,
        partition_plan: PartitionPlan | None = None,
        partitioning: BioSievePartitionConfig | None = None,
        biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
        preprocessing: PreprocessingConfig | Any | None = None,
        search_space: SearchSpace | None = None,
        evaluation_role: EvaluationRole = "auto",
        require_complete: bool = True,
        model_params: Mapping[str, Any] | None = None,
    ) -> OptimizationResult:
        spec = self.registry.get(algorithm)
        dataset.validate(task=spec.task)

        plan = resolve_partition_plan(
            dataset=dataset,
            partition_plan=partition_plan,
            partitioning=partitioning,
            biosieve_extra_columns=biosieve_extra_columns,
        )
        plan.validate_against(dataset, require_complete=require_complete)

        search_dataset, cv, evaluation_roles = build_explicit_cv(
            dataset=dataset,
            plan=plan,
            evaluation_role=evaluation_role,
            require_complete=require_complete,
        )

        for split_index, (train_index, _) in enumerate(cv):
            train_ids = tuple(search_dataset.sample_ids[index] for index in train_index)
            try:
                search_dataset.subset(train_ids).validate(task=spec.task)
            except DatasetValidationError as exc:
                raise ValidationContractError(
                    f"Training membership for tuning split {split_index} is invalid "
                    f"for task '{spec.task}': {exc}"
                ) from exc

        space = search_space if search_space is not None else spec.get_search_space()
        if space is None:
            raise ValidationContractError(f"No search space defined for algorithm '{spec.name}'.")
        if len(space) == 0:
            raise ValidationContractError(f"Search space for algorithm '{spec.name}' is empty.")

        metrics = tuple(dict.fromkeys(config.metrics))
        if not metrics:
            raise ValidationContractError("At least one tuning metric is required.")
        refit_metric = config.resolved_refit_metric()
        scorers = {metric: get_scorer(metric, task=spec.task, y=search_dataset.y) for metric in metrics}

        first_train_index = cv[0][0]
        first_train = search_dataset.subset(
            tuple(search_dataset.sample_ids[index] for index in first_train_index)
        )
        estimator = spec.build_estimator(
            random_state=config.random_state,
            **dict(model_params or {}),
        )
        pipeline = build_model_pipeline(
            spec=spec,
            estimator=estimator,
            training_data=first_train,
            preprocessing=preprocessing,
        )

        fit_params = _fit_params(search_dataset, spec)
        start = perf_counter()

        if config.optimizer == "optuna":
            result = self._run_optuna(
                spec=spec,
                pipeline=pipeline,
                dataset=search_dataset,
                cv=cv,
                search_space=space,
                scorers=scorers,
                refit_metric=refit_metric,
                config=config,
                fit_params=fit_params,
            )
        else:
            result = self._run_sklearn(
                spec=spec,
                pipeline=pipeline,
                dataset=search_dataset,
                cv=cv,
                search_space=space,
                scorers=scorers,
                refit_metric=refit_metric,
                config=config,
                fit_params=fit_params,
            )

        result.partition_plan = plan
        result.metrics = metrics
        result.refit_metric = refit_metric
        result.metadata.update(
            {
                "dataset_fingerprint": dataset.fingerprint,
                "search_dataset_fingerprint": search_dataset.fingerprint,
                "partition_fingerprint": plan.fingerprint,
                "partition_source": plan.metadata.get("source", "external"),
                "evaluation_roles": evaluation_roles,
                "n_splits": len(cv),
                "n_search_samples": search_dataset.n_samples,
                "n_total_samples": dataset.n_samples,
                "protected_samples": dataset.n_samples - search_dataset.n_samples,
                "elapsed_seconds": float(perf_counter() - start),
                "search_space": space.to_dict(),
            }
        )
        return result

    def _run_sklearn(
        self,
        *,
        spec: Any,
        pipeline: Any,
        dataset: DatasetBundle,
        cv: tuple[tuple[np.ndarray, np.ndarray], ...],
        search_space: SearchSpace,
        scorers: dict[str, Any],
        refit_metric: str,
        config: TuningConfig,
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

        try:
            grid_parameters = (
                search_space.to_grid(prefix="estimator__") if optimizer in {"grid", "halving_grid"} else None
            )
            random_parameters = (
                search_space.to_random(prefix="estimator__")
                if optimizer in {"random", "halving_random"}
                else None
            )
        except ValueError as exc:
            raise ValidationContractError(
                f"Search space for optimizer '{optimizer}' is incompatible: {exc}"
            ) from exc

        if optimizer == "grid":
            search = GridSearchCV(
                param_grid=grid_parameters,
                **common,
            )
        elif optimizer == "random":
            search = RandomizedSearchCV(
                param_distributions=random_parameters,
                n_iter=config.n_iter,
                random_state=config.random_state,
                **common,
            )
        elif optimizer in {"halving_grid", "halving_random"}:
            # sklearn's successive-halving estimators currently accept only a
            # single scoring objective. We optimize the explicit refit metric,
            # then evaluate the selected configuration with all requested
            # metrics on the same explicit folds.
            single_common = dict(common)
            single_common["scoring"] = scorers[refit_metric]
            single_common["refit"] = config.refit

            from sklearn.experimental import enable_halving_search_cv  # noqa: F401
            from sklearn.model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

            halving_common = {
                **single_common,
                "factor": config.factor,
                "resource": config.resource,
                "max_resources": config.max_resources,
                "min_resources": config.min_resources,
                "aggressive_elimination": config.aggressive_elimination,
            }
            if optimizer == "halving_grid":
                search = HalvingGridSearchCV(
                    param_grid=grid_parameters,
                    **halving_common,
                )
            else:
                min_resources = config.min_resources
                if min_resources == "exhaust":
                    # HalvingRandomSearchCV uses "smallest" as its canonical
                    # automatic lower-resource policy.
                    min_resources = "smallest"
                halving_common["min_resources"] = min_resources
                search = HalvingRandomSearchCV(
                    param_distributions=random_parameters,
                    random_state=config.random_state,
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
            history = _history_from_multimetric_results(results, tuple(scorers))
        else:
            best_index = _best_index(
                results["mean_test_score"],
                spec.name,
                refit_metric,
                optimizer,
            )
            best_score = float(results["mean_test_score"][best_index])
            history = _history_from_singlemetric_results(results, refit_metric)
            best_scores = _evaluate_selected_metrics(
                pipeline=search.estimator,
                params=results["params"][best_index],
                dataset=dataset,
                cv=cv,
                scorers=scorers,
                fit_params=fit_params,
                n_jobs=config.n_jobs,
            )
            best_scores[refit_metric] = best_score

        best_score = _ensure_finite(
            best_score,
            algorithm=spec.name,
            metric=refit_metric,
            optimizer=optimizer,
        )
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
            metric=refit_metric,
            best_score=best_score,
            best_params=best_params,
            best_model=best_model,
            spec=spec,
            history=history,
            study=search,
            refit=config.refit,
            best_scores=best_scores,
        )

    def _run_optuna(
        self,
        *,
        spec: Any,
        pipeline: Any,
        dataset: DatasetBundle,
        cv: tuple[tuple[np.ndarray, np.ndarray], ...],
        search_space: SearchSpace,
        scorers: dict[str, Any],
        refit_metric: str,
        config: TuningConfig,
        fit_params: dict[str, Any],
    ) -> OptimizationResult:
        try:
            import optuna
            from optuna.trial import TrialState
        except ImportError as exc:
            from saber.exceptions import OptionalDependencyError

            raise OptionalDependencyError(
                dependency="optuna",
                extra="optuna",
                purpose="Optuna hyperparameter optimization",
            ) from exc

        sampler = None
        sampler_seed = config.random_state
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
                    params=fit_params or None,
                    error_score=config.error_score,
                )["test_score"]
                score = float(np.mean(np.asarray(scores, dtype=float)))
                if not np.isfinite(score):
                    raise NonFiniteScoreError(
                        algorithm=spec.name,
                        metric=refit_metric,
                        optimizer="optuna",
                        score=score,
                    )
                return score
            except Exception as exc:
                trial.set_user_attr("saber_error", f"{type(exc).__name__}: {exc}")
                raise

        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout,
            catch=(Exception,),
        )

        completed = [
            trial
            for trial in study.trials
            if trial.state == TrialState.COMPLETE
            and trial.value is not None
            and np.isfinite(float(trial.value))
        ]
        if not completed:
            raise NonFiniteScoreError(
                algorithm=spec.name,
                metric=refit_metric,
                optimizer="optuna",
                score=float("nan"),
            )

        best_trial = max(completed, key=lambda trial: float(trial.value))
        best_score = _ensure_finite(
            float(best_trial.value),
            algorithm=spec.name,
            metric=refit_metric,
            optimizer="optuna",
        )
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
            history.append(
                {
                    "trial": trial.number,
                    "params": dict(trial.params),
                    "score": None if trial.value is None else float(trial.value),
                    "metrics": {refit_metric: None if trial.value is None else float(trial.value)},
                    "status": str(trial.state).split(".")[-1].lower(),
                    "error": trial.user_attrs.get("saber_error"),
                }
            )

        return OptimizationResult(
            algorithm=spec.name,
            optimizer="optuna",
            metric=refit_metric,
            best_score=best_score,
            best_params=best_params,
            best_model=best_model,
            spec=spec,
            history=history,
            study=study,
            refit=config.refit,
            best_scores=best_scores,
        )


def tune_model(registry: AlgorithmRegistry, **kwargs: Any) -> OptimizationResult:
    """Functional convenience wrapper around :class:`TuningEngine`."""
    return TuningEngine(registry).run(**kwargs)


def _fit_params(dataset: DatasetBundle, spec: Any) -> dict[str, Any]:
    if dataset.sample_weight is None:
        return {}
    if not spec.capabilities.sample_weight:
        raise ValidationContractError(
            f"Dataset supplies sample_weight but algorithm '{spec.name}' "
            "does not advertise sample-weight support."
        )
    return {
        "estimator__sample_weight": np.asarray(dataset.sample_weight, dtype=float),
    }


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


def _ensure_finite(score: float, *, algorithm: str, metric: str, optimizer: str) -> float:
    value = float(score)
    if not np.isfinite(value):
        raise NonFiniteScoreError(
            algorithm=algorithm,
            metric=metric,
            optimizer=optimizer,
            score=value,
        )
    return value


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


def _strip_estimator_prefix(params: dict[str, Any]) -> dict[str, Any]:
    prefix = "estimator__"
    return {key.removeprefix(prefix): value for key, value in params.items()}


def _candidate_status(value: float) -> str:
    return "complete" if np.isfinite(float(value)) else "failed"


def _history_from_multimetric_results(
    results: Mapping[str, Any],
    metrics: tuple[str, ...],
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, raw_params in enumerate(results["params"]):
        metric_payload = {
            metric: {
                "mean": float(results[f"mean_test_{metric}"][index]),
                "std": float(results[f"std_test_{metric}"][index]),
                "rank": int(results[f"rank_test_{metric}"][index]),
            }
            for metric in metrics
        }
        finite = all(np.isfinite(payload["mean"]) for payload in metric_payload.values())
        history.append(
            {
                "params": _strip_estimator_prefix(dict(raw_params)),
                "metrics": metric_payload,
                "status": "complete" if finite else "failed",
                "error": None if finite else "non-finite candidate score",
            }
        )
    return history


def _history_from_singlemetric_results(
    results: Mapping[str, Any],
    metric: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for index, raw_params in enumerate(results["params"]):
        mean = float(results["mean_test_score"][index])
        history.append(
            {
                "params": _strip_estimator_prefix(dict(raw_params)),
                "metrics": {
                    metric: {
                        "mean": mean,
                        "std": float(results["std_test_score"][index]),
                        "rank": int(results["rank_test_score"][index]),
                    }
                },
                "status": _candidate_status(mean),
                "error": None if np.isfinite(mean) else "non-finite candidate score",
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
        params=dict(fit_params) or None,
        error_score=np.nan,
    )
    output: dict[str, float] = {}
    for metric in scorers:
        values = np.asarray(scores[f"test_{metric}"], dtype=float)
        output[metric] = float(np.mean(values))
    return output
