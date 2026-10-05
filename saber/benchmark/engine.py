"""Systematic benchmark orchestration over validation and tuning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from saber.benchmark.config import BenchmarkConfig
from saber.benchmark.results import BenchmarkResult, BenchmarkRun
from saber.core.registry import get_algorithm
from saber.datasets import BioSievePartitionConfig, DatasetBundle, PartitionPlan
from saber.exceptions import BenchmarkContractError
from saber.tuning import tune
from saber.utils.tabular import python_scalar
from saber.validation import validate
from saber.validation.partitioning import resolve_partition_plan

if TYPE_CHECKING:
    from collections.abc import Sequence

    from saber.core.specs import AlgorithmSpec
    from saber.preprocessing import PreprocessingConfig
    from saber.tuning.results import OptimizationResult
    from saber.validation.partitioning import EvaluationRole
    from saber.validation.results import ValidationResult


def benchmark(
    *,
    datasets: DatasetBundle | Mapping[str, DatasetBundle],
    algorithms: Sequence[str],
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    preprocessing: PreprocessingConfig | None = None,
    partitions: PartitionPlan | BioSievePartitionConfig | Mapping[str, PartitionPlan] | None = None,
    config: BenchmarkConfig | None = None,
    search_spaces: Mapping[str, Any] | None = None,
    metrics: Sequence[str],
    evaluation_role: EvaluationRole = "auto",
    require_complete: bool = True,
    positive_class: Any | None = None,
    return_estimators: bool = False,
) -> BenchmarkResult:
    """Run the full algorithm x representation x partition x seed x mode matrix.

    ``datasets`` is one prepared dataset or a mapping of representation label
    to dataset (all with the same sample IDs and targets).  ``partitions`` is
    one plan, a mapping of scenario label to plan, or a
    :class:`BioSievePartitionConfig` that generates one plan from the first
    dataset.  Each seed in ``BenchmarkConfig.seeds`` is the ``random_state`` of
    one set of runs.
    """
    config = BenchmarkConfig() if config is None else config
    metrics = tuple(dict.fromkeys(metrics))
    if not metrics:
        raise BenchmarkContractError("benchmark metrics must contain at least one metric.")
    benchmark_datasets = _normalize_datasets(datasets)
    _validate_dataset_alignment(benchmark_datasets)
    algorithm_names = tuple(dict.fromkeys(algorithms))
    if not algorithm_names and not config.include_baselines:
        raise BenchmarkContractError("At least one benchmark algorithm is required.")

    task = _infer_benchmark_task(algorithm_names)
    if task is None:
        raise BenchmarkContractError("Cannot infer benchmark task from the requested algorithms.")

    baseline_name = "dummy_classifier" if task == "classification" else "dummy_regressor"
    if config.include_baselines and baseline_name not in algorithm_names:
        algorithm_names = (baseline_name, *algorithm_names)

    partition_scenarios = _resolve_partitions(
        datasets=benchmark_datasets,
        partitions=partitions,
        require_complete=require_complete,
    )

    options = _RunOptions(
        config=config,
        metrics=metrics,
        evaluation_role=evaluation_role,
        require_complete=require_complete,
        positive_class=positive_class,
        return_estimators=return_estimators,
        preprocessing=preprocessing,
    )
    params_by_algorithm = dict(model_params or {})
    spaces_by_algorithm = dict(search_spaces or {})
    runs: list[BenchmarkRun] = []

    for label, dataset in benchmark_datasets.items():
        for partition in partition_scenarios:
            bound_plan = _bind_partition_plan(
                partition.plan,
                dataset,
                require_complete=require_complete,
            )
            for algorithm in algorithm_names:
                spec = _safe_get_spec(algorithm)
                run_modes = ("baseline",) if _is_baseline(spec) else config.modes
                for seed in config.seeds:
                    for mode in run_modes:
                        run = _run_one(
                            label=label,
                            dataset=dataset,
                            partition=partition,
                            bound_plan=bound_plan,
                            algorithm=algorithm,
                            algorithm_spec=spec,
                            mode=mode,
                            seed=seed,
                            options=options,
                            model_params=dict(params_by_algorithm.get(algorithm, {})),
                            search_space=spaces_by_algorithm.get(algorithm),
                        )
                        runs.append(run)
                        if run.status == "failed" and config.fail_fast:
                            raise BenchmarkContractError(run.error or "Benchmark run failed.")

    return BenchmarkResult(
        runs=tuple(runs),
        targets=_targets_by_id(next(iter(benchmark_datasets.values()))),
        metadata={
            "n_datasets": len(benchmark_datasets),
            "n_partition_scenarios": len(partition_scenarios),
            "algorithms": tuple(algorithm_names),
            "seeds": config.seeds,
            "modes": config.modes,
            "metrics": metrics,
            "task": task,
            **dict(config.metadata),
        },
    )


@dataclass(frozen=True, slots=True)
class _Partition:
    """One named partition scenario of the benchmark matrix."""

    label: str
    plan: PartitionPlan


@dataclass(frozen=True, slots=True)
class _RunOptions:
    """Knobs shared by every run of one benchmark call."""

    config: BenchmarkConfig
    metrics: tuple[str, ...]
    evaluation_role: EvaluationRole
    require_complete: bool
    positive_class: Any | None
    return_estimators: bool
    preprocessing: PreprocessingConfig | None


def _resolve_partitions(
    *,
    datasets: dict[str, DatasetBundle],
    partitions: PartitionPlan | BioSievePartitionConfig | Mapping[str, PartitionPlan] | None,
    require_complete: bool,
) -> tuple[_Partition, ...]:
    if partitions is None:
        raise BenchmarkContractError(
            "Unpartitioned benchmark data require a PartitionPlan or BioSievePartitionConfig."
        )
    reference = next(iter(datasets.values()))
    if isinstance(partitions, BioSievePartitionConfig):
        plan = resolve_partition_plan(dataset=reference, partition_plan=partitions)
        plan.validate_against(reference, require_complete=require_complete)
        label = str(plan.metadata.get("strategy") or partitions.strategy)
        return (_Partition(label=label, plan=plan),)

    normalized = _normalize_partitions(partitions)
    known_fingerprints = {dataset.fingerprint for dataset in datasets.values()}
    for item in normalized:
        if (
            item.plan.dataset_fingerprint is not None
            and item.plan.dataset_fingerprint not in known_fingerprints
        ):
            raise BenchmarkContractError(
                f"Partition '{item.label}' targets a dataset fingerprint that is not "
                "one of the benchmark representations. Use a fingerprint-free external "
                "membership plan or a plan generated from one benchmark dataset."
            )
        _bind_partition_plan(item.plan, reference, require_complete=require_complete)
    return normalized


def _run_one(
    *,
    label: str,
    dataset: DatasetBundle,
    partition: _Partition,
    bound_plan: PartitionPlan,
    algorithm: str,
    algorithm_spec: AlgorithmSpec | None,
    mode: str,
    seed: int | None,
    options: _RunOptions,
    model_params: dict[str, Any],
    search_space: Any | None,
) -> BenchmarkRun:
    run_id = _run_id(
        label,
        partition.label,
        algorithm,
        mode,
        seed,
    )
    start = perf_counter()
    provider = None if algorithm_spec is None else algorithm_spec.provider
    task = None if algorithm_spec is None else algorithm_spec.task
    identity = {
        "run_id": run_id,
        "representation": label,
        "partition_label": partition.label,
        "algorithm": algorithm,
        "provider": provider,
        "task": task,
        "mode": mode,
        "seed": seed,
    }
    fingerprints = {
        "dataset_fingerprint": dataset.fingerprint,
        "partition_fingerprint": bound_plan.fingerprint,
    }

    try:
        if algorithm_spec is None:
            algorithm_spec = get_algorithm(algorithm)  # raises AlgorithmNotFoundError, recorded below
        if task is not None:
            dataset.validate(task=task)

        if mode in {"untuned", "baseline"}:
            # When tuned and untuned modes are compared in the same
            # holdout benchmark, every reported score must come from the
            # same protected final test.  Otherwise ``auto`` would score
            # untuned runs on validation while tuned runs report test,
            # producing an apples-to-oranges leaderboard.  Refit the
            # untuned/baseline model on train+validation and evaluate the
            # protected test, matching the final tuned evaluation path.
            mixed_with_tuned = "tuned" in options.config.modes
            if mixed_with_tuned and _has_protected_test(bound_plan):
                evaluation_plan = _protected_test_plan(bound_plan, dataset)
                evaluation_role = "test"
            else:
                evaluation_plan = bound_plan
                evaluation_role = options.evaluation_role

            validation = validate(
                dataset=dataset,
                algorithm=algorithm,
                model_params=model_params,
                preprocessing=options.preprocessing,
                partition_plan=evaluation_plan,
                metrics=options.metrics,
                evaluation_role=evaluation_role,
                require_complete=options.require_complete,
                positive_class=options.positive_class,
                return_estimators=options.return_estimators,
                random_state=seed,
            )
            optimization = None
        elif mode == "tuned":
            validation, optimization = _run_tuned(
                dataset=dataset,
                algorithm=algorithm,
                plan=bound_plan,
                options=options,
                model_params=model_params,
                search_space=search_space,
                seed=seed,
            )
        else:
            # caught below so a bad mode is recorded as a failed run, not a crash
            raise BenchmarkContractError(f"Unsupported benchmark mode '{mode}'.")  # noqa: TRY301

        elapsed = perf_counter() - start
        best_params = {} if optimization is None else optimization.best_params
        return BenchmarkRun(
            **identity,
            status="complete",
            validation=validation,
            optimization=optimization,
            elapsed_seconds=float(elapsed),
            parameters={**algorithm_spec.default_params, **model_params, **best_params},
            metadata=fingerprints,
        )
    except Exception as exc:  # noqa: BLE001  # any run failure becomes a failed BenchmarkRun, not a crash
        default_params = {} if algorithm_spec is None else algorithm_spec.default_params
        return BenchmarkRun(
            **identity,
            status="failed",
            elapsed_seconds=float(perf_counter() - start),
            error=f"{type(exc).__name__}: {exc}",
            parameters={**default_params, **model_params},
            metadata=fingerprints,
        )


def _run_tuned(
    *,
    dataset: DatasetBundle,
    algorithm: str,
    plan: PartitionPlan,
    options: _RunOptions,
    model_params: dict[str, Any],
    search_space: Any | None,
    seed: int | None,
) -> tuple[ValidationResult, OptimizationResult]:
    final_plan = _protected_test_plan(plan, dataset)
    if options.config.tuning is None:
        raise BenchmarkContractError("Tuned benchmark requires TuningConfig.")
    optimization = tune(
        dataset=dataset,
        algorithm=algorithm,
        model_params=model_params,
        preprocessing=options.preprocessing,
        partition_plan=plan,
        config=options.config.tuning,
        search_space=search_space,
        metrics=options.metrics,
        evaluation_role="auto",
        require_complete=options.require_complete,
        positive_class=options.positive_class,
        random_state=seed,
    )
    final_params = {**model_params, **optimization.best_params}
    validation = validate(
        dataset=dataset,
        algorithm=algorithm,
        model_params=final_params,
        preprocessing=options.preprocessing,
        partition_plan=final_plan,
        metrics=options.metrics,
        evaluation_role="test",
        require_complete=options.require_complete,
        positive_class=options.positive_class,
        return_estimators=options.return_estimators,
        random_state=seed,
    )
    validation.metadata.update(
        {
            "benchmark_tuned": True,
            "tuning_partition_fingerprint": plan.fingerprint,
            "tuning_refit_metric": optimization.refit_metric,
        }
    )
    return validation, optimization


def _normalize_datasets(
    datasets: DatasetBundle | Mapping[str, DatasetBundle],
) -> dict[str, DatasetBundle]:
    if isinstance(datasets, DatasetBundle):
        return {"dataset": datasets}
    if not isinstance(datasets, Mapping):
        raise BenchmarkContractError(
            "datasets must be a DatasetBundle or a mapping of label to DatasetBundle."
        )
    result = {str(label): dataset for label, dataset in datasets.items()}
    if not result:
        raise BenchmarkContractError("At least one benchmark dataset is required.")
    if any(not label.strip() for label in result):
        raise BenchmarkContractError("Benchmark dataset labels cannot be empty.")
    if any(not isinstance(dataset, DatasetBundle) for dataset in result.values()):
        raise BenchmarkContractError("Benchmark dataset mappings must contain DatasetBundle values.")
    return result


def _normalize_partitions(
    partitions: PartitionPlan | Mapping[str, PartitionPlan],
) -> tuple[_Partition, ...]:
    if isinstance(partitions, PartitionPlan):
        return (_Partition(label="default", plan=partitions),)
    if not isinstance(partitions, Mapping):
        raise BenchmarkContractError(
            "partitions must be a PartitionPlan, a mapping of label to PartitionPlan, "
            "or a BioSievePartitionConfig."
        )
    result = tuple(_Partition(label=str(label), plan=plan) for label, plan in partitions.items())
    if not result:
        raise BenchmarkContractError("At least one benchmark partition is required.")
    if any(not item.label.strip() for item in result):
        raise BenchmarkContractError("Benchmark partition labels cannot be empty.")
    if any(not isinstance(item.plan, PartitionPlan) for item in result):
        raise BenchmarkContractError("Benchmark partition mappings must contain PartitionPlan values.")
    return result


def _validate_dataset_alignment(datasets: dict[str, DatasetBundle]) -> None:
    reference, *others = datasets.values()
    reference_targets = _targets_by_id(reference)
    reference_ids = set(reference_targets)
    for dataset in others:
        current_targets = _targets_by_id(dataset)
        if set(current_targets) != reference_ids:
            raise BenchmarkContractError("Representation datasets must contain the same sample IDs.")
        for sample_id, target in reference_targets.items():
            if not _target_equal(target, current_targets[sample_id]):
                raise BenchmarkContractError(
                    "Representation datasets must contain identical targets for each sample ID."
                )


def _targets_by_id(dataset: DatasetBundle) -> dict[Any, Any]:
    return {
        sample_id: python_scalar(target)
        for sample_id, target in zip(dataset.resolved_sample_ids, np.asarray(dataset.y), strict=True)
    }


def _target_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float) and np.isnan(left) and np.isnan(right):
        return True
    return bool(left == right)


def _infer_benchmark_task(algorithms: tuple[str, ...]) -> str | None:
    tasks = []
    for name in algorithms:
        try:
            task = get_algorithm(name).task
        except Exception:  # noqa: BLE001, S112  # unresolvable algorithms are simply excluded from task inference
            continue
        if task not in tasks:
            tasks.append(task)
    if len(tasks) > 1:
        raise BenchmarkContractError(f"A benchmark matrix cannot mix supervised tasks: {tasks!r}.")
    return tasks[0] if tasks else None


def _safe_get_spec(algorithm: str) -> AlgorithmSpec | None:
    try:
        return get_algorithm(algorithm)
    except Exception:  # noqa: BLE001  # unknown/invalid algorithm names resolve to None, not a crash
        return None


def _is_baseline(spec: Any | None) -> bool:
    return spec is not None and "baseline" in spec.tags


def _bind_partition_plan(
    plan: PartitionPlan,
    dataset: DatasetBundle,
    *,
    require_complete: bool,
) -> PartitionPlan:
    """Safely bind membership-identical plans across prepared representations."""
    if plan.dataset_fingerprint in {None, dataset.fingerprint}:
        plan.validate_against(dataset, require_complete=require_complete)
        return plan

    rebound = PartitionPlan(
        kind=plan.kind,
        splits=plan.splits,
        dataset_fingerprint=dataset.fingerprint,
        metadata={
            **dict(plan.metadata),
            "benchmark_rebound_from_dataset_fingerprint": plan.dataset_fingerprint,
        },
    )
    rebound.validate_against(dataset, require_complete=require_complete)
    return rebound


def _has_protected_test(plan: PartitionPlan) -> bool:
    """Whether a plan is the holdout shape required for final test reporting."""
    if plan.kind != "holdout" or len(plan.splits) != 1:
        return False
    split = plan.splits[0]
    return bool(split.validation_ids and split.test_ids)


def _protected_test_plan(plan: PartitionPlan, dataset: DatasetBundle) -> PartitionPlan:
    """Create a final train+validation → test plan for unbiased tuned reporting."""
    if plan.kind != "holdout" or len(plan.splits) != 1:
        raise BenchmarkContractError(
            "Tuned benchmark reporting requires one holdout split with distinct "
            "train, validation, and protected test memberships. CV tuning must use "
            "nested CV or an external final test set."
        )
    split = plan.splits[0]
    if not split.validation_ids or not split.test_ids:
        raise BenchmarkContractError(
            "Tuned benchmark reporting requires both validation and protected test memberships."
        )
    development_ids = tuple(split.train_ids) + tuple(split.validation_ids)
    return PartitionPlan.holdout(
        train_ids=development_ids,
        test_ids=split.test_ids,
        dataset=dataset,
        name=f"{split.name}_final_test",
        metadata={
            **dict(plan.metadata),
            "benchmark_final_test": True,
            "source_partition_fingerprint": plan.fingerprint,
        },
    )


def _run_id(dataset: str, partition: str, algorithm: str, mode: str, seed: int | None) -> str:
    payload = "|".join((dataset, partition, algorithm, mode, repr(seed)))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
