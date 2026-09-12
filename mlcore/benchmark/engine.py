"""Systematic benchmark orchestration over validation and tuning engines."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np

from mlcore.benchmark.config import BenchmarkConfig
from mlcore.benchmark.results import BenchmarkResult, BenchmarkRun
from mlcore.benchmark.specs import BenchmarkDataset, BenchmarkPartition
from mlcore.core.registry import AlgorithmRegistry
from mlcore.datasets import DatasetBundle, PartitionPlan, PartitionSplit
from mlcore.datasets.biosieve import BioSievePartitionConfig
from mlcore.exceptions import BenchmarkContractError
from mlcore.preprocessing import PreprocessingConfig
from mlcore.tuning import TuningConfig, TuningEngine
from mlcore.validation import ValidationEngine
from mlcore.validation.partitioning import resolve_partition_plan


class BenchmarkEngine:
    """Orchestrate reproducible classical supervised benchmark matrices.

    The engine intentionally delegates all model evaluation to
    :class:`ValidationEngine` and all hyperparameter search to
    :class:`TuningEngine`.  It does not implement another training, splitting,
    preprocessing, or scoring path.
    """

    def __init__(self, registry: AlgorithmRegistry) -> None:
        self.registry = registry
        self.validation_engine = ValidationEngine(registry)
        self.tuning_engine = TuningEngine(registry)

    def run(
        self,
        *,
        datasets: DatasetBundle | BenchmarkDataset | Mapping[str, DatasetBundle] | Sequence[BenchmarkDataset],
        algorithms: Sequence[str],
        config: BenchmarkConfig,
        partitions: PartitionPlan | BenchmarkPartition | Mapping[str, PartitionPlan] | Sequence[BenchmarkPartition] | None = None,
        partitioning: BioSievePartitionConfig | None = None,
        partitioning_reference: str | None = None,
        biosieve_extra_columns: Mapping[str, Sequence[Any]] | None = None,
        preprocessing: PreprocessingConfig | Any | None = None,
        model_params: Mapping[str, Mapping[str, Any]] | None = None,
        search_spaces: Mapping[str, Any] | None = None,
        positive_class: Any | None = None,
    ) -> BenchmarkResult:
        benchmark_datasets = _normalize_datasets(datasets)
        _validate_dataset_alignment(benchmark_datasets)
        algorithm_names = tuple(dict.fromkeys(algorithms))
        if not algorithm_names and not config.include_baselines:
            raise BenchmarkContractError("At least one benchmark algorithm is required.")

        task = _infer_benchmark_task(self.registry, algorithm_names)
        if task is None:
            raise BenchmarkContractError(
                "Cannot infer benchmark task from the requested algorithms."
            )

        baseline_name = "dummy_classifier" if task == "classification" else "dummy_regressor"
        if config.include_baselines and baseline_name not in algorithm_names:
            algorithm_names = (baseline_name, *algorithm_names)

        partition_scenarios = self._resolve_partitions(
            datasets=benchmark_datasets,
            partitions=partitions,
            partitioning=partitioning,
            partitioning_reference=partitioning_reference,
            biosieve_extra_columns=biosieve_extra_columns,
            require_complete=config.require_complete,
        )

        params_by_algorithm = dict(model_params or {})
        spaces_by_algorithm = dict(search_spaces or {})
        runs: list[BenchmarkRun] = []

        for dataset_spec in benchmark_datasets:
            for partition_spec in partition_scenarios:
                bound_plan = _bind_partition_plan(
                    partition_spec.plan,
                    dataset_spec.dataset,
                    require_complete=config.require_complete,
                )
                for algorithm in algorithm_names:
                    spec = _safe_get_spec(self.registry, algorithm)
                    run_modes = ("baseline",) if _is_baseline(spec, algorithm) else config.modes
                    for seed in config.seeds:
                        for mode in run_modes:
                            run = self._run_one(
                                dataset_spec=dataset_spec,
                                partition_spec=partition_spec,
                                bound_plan=bound_plan,
                                algorithm=algorithm,
                                algorithm_spec=spec,
                                mode=mode,
                                seed=seed,
                                config=config,
                                preprocessing=preprocessing,
                                model_params=dict(params_by_algorithm.get(algorithm, {})),
                                search_space=spaces_by_algorithm.get(algorithm),
                                positive_class=positive_class,
                            )
                            runs.append(run)
                            if run.status == "failed" and config.fail_fast:
                                raise BenchmarkContractError(run.error or "Benchmark run failed.")

        return BenchmarkResult(
            runs=tuple(runs),
            metadata={
                "n_datasets": len(benchmark_datasets),
                "n_partition_scenarios": len(partition_scenarios),
                "algorithms": tuple(algorithm_names),
                "seeds": config.seeds,
                "modes": config.modes,
                "metrics": config.metrics,
                "task": task,
                **dict(config.metadata),
            },
        )

    def _resolve_partitions(
        self,
        *,
        datasets: tuple[BenchmarkDataset, ...],
        partitions: PartitionPlan | BenchmarkPartition | Mapping[str, PartitionPlan] | Sequence[BenchmarkPartition] | None,
        partitioning: BioSievePartitionConfig | None,
        partitioning_reference: str | None,
        biosieve_extra_columns: Mapping[str, Sequence[Any]] | None,
        require_complete: bool,
    ) -> tuple[BenchmarkPartition, ...]:
        if partitions is not None and partitioning is not None:
            raise BenchmarkContractError(
                "Provide explicit benchmark partitions or BioSieve partitioning, not both."
            )
        if partitions is not None:
            normalized = _normalize_partitions(partitions)
            known_fingerprints = {item.dataset.fingerprint for item in datasets}
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
                _bind_partition_plan(
                    item.plan,
                    datasets[0].dataset,
                    require_complete=require_complete,
                )
            return normalized
        if partitioning is None:
            raise BenchmarkContractError(
                "Unpartitioned benchmark data require BioSievePartitionConfig."
            )

        reference = _partition_reference(datasets, partitioning_reference)
        plan = resolve_partition_plan(
            dataset=reference.dataset,
            partition_plan=None,
            partitioning=partitioning,
            biosieve_extra_columns=biosieve_extra_columns,
        )
        plan.validate_against(reference.dataset, require_complete=require_complete)
        label = str(plan.metadata.get("strategy") or partitioning.strategy)
        metadata = {
            "source": "biosieve",
            "partitioning_reference": reference.label,
        }
        return (BenchmarkPartition(label=label, plan=plan, metadata=metadata),)

    def _run_one(
        self,
        *,
        dataset_spec: BenchmarkDataset,
        partition_spec: BenchmarkPartition,
        bound_plan: PartitionPlan,
        algorithm: str,
        algorithm_spec: Any | None,
        mode: str,
        seed: int | None,
        config: BenchmarkConfig,
        preprocessing: PreprocessingConfig | Any | None,
        model_params: dict[str, Any],
        search_space: Any | None,
        positive_class: Any | None,
    ) -> BenchmarkRun:
        run_id = _run_id(
            dataset_spec.label,
            partition_spec.label,
            algorithm,
            mode,
            seed,
        )
        start = perf_counter()
        provider = None if algorithm_spec is None else algorithm_spec.provider
        task = None if algorithm_spec is None else algorithm_spec.task
        targets = _targets_by_id(dataset_spec.dataset)

        try:
            if algorithm_spec is None:
                self.registry.get(algorithm)  # raises canonical registry error
            if task is not None:
                dataset_spec.dataset.validate(task=task)

            if mode in {"untuned", "baseline"}:
                # When tuned and untuned modes are compared in the same
                # holdout benchmark, every reported score must come from the
                # same protected final test.  Otherwise ``auto`` would score
                # untuned runs on validation while tuned runs report test,
                # producing an apples-to-oranges leaderboard.  Refit the
                # untuned/baseline model on train+validation and evaluate the
                # protected test, matching the final tuned evaluation path.
                mixed_with_tuned = "tuned" in config.modes
                if mixed_with_tuned and _has_protected_test(bound_plan):
                    evaluation_plan = _protected_test_plan(bound_plan, dataset_spec.dataset)
                    evaluation_role = "test"
                else:
                    evaluation_plan = bound_plan
                    evaluation_role = config.evaluation_role

                validation = self.validation_engine.run(
                    dataset=dataset_spec.dataset,
                    algorithm=algorithm,
                    partition_plan=evaluation_plan,
                    preprocessing=preprocessing,
                    metrics=config.metrics,
                    evaluation_role=evaluation_role,
                    positive_class=positive_class,
                    random_state=seed,
                    return_estimators=config.return_estimators,
                    require_complete=config.require_complete,
                    **model_params,
                )
                optimization = None
            elif mode == "tuned":
                validation, optimization = self._run_tuned(
                    dataset=dataset_spec.dataset,
                    algorithm=algorithm,
                    plan=bound_plan,
                    config=config,
                    preprocessing=preprocessing,
                    model_params=model_params,
                    search_space=search_space,
                    positive_class=positive_class,
                    seed=seed,
                )
            else:
                raise BenchmarkContractError(f"Unsupported benchmark mode '{mode}'.")

            elapsed = perf_counter() - start
            return BenchmarkRun(
                run_id=run_id,
                dataset_label=dataset_spec.label,
                representation=dataset_spec.representation_label,
                partition_label=partition_spec.label,
                algorithm=algorithm,
                provider=provider,
                task=task,
                mode=mode,
                seed=seed,
                status="complete",
                validation=validation,
                optimization=optimization,
                elapsed_seconds=float(elapsed),
                parameters=(
                    {**dict(algorithm_spec.default_params), **model_params, **optimization.best_params}
                    if optimization is not None
                    else {**dict(algorithm_spec.default_params), **model_params}
                ),
                targets=targets,
                metadata={
                    "dataset_fingerprint": dataset_spec.dataset.fingerprint,
                    "partition_fingerprint": bound_plan.fingerprint,
                    "representation_metadata": dict(dataset_spec.metadata),
                    "partition_metadata": dict(partition_spec.metadata),
                },
            )
        except Exception as exc:
            return BenchmarkRun(
                run_id=run_id,
                dataset_label=dataset_spec.label,
                representation=dataset_spec.representation_label,
                partition_label=partition_spec.label,
                algorithm=algorithm,
                provider=provider,
                task=task,
                mode=mode,
                seed=seed,
                status="failed",
                elapsed_seconds=float(perf_counter() - start),
                error=f"{type(exc).__name__}: {exc}",
                parameters=(
                    {**dict(algorithm_spec.default_params), **model_params}
                    if algorithm_spec is not None
                    else dict(model_params)
                ),
                targets=targets,
                metadata={
                    "dataset_fingerprint": dataset_spec.dataset.fingerprint,
                    "partition_fingerprint": bound_plan.fingerprint,
                },
            )

    def _run_tuned(
        self,
        *,
        dataset: DatasetBundle,
        algorithm: str,
        plan: PartitionPlan,
        config: BenchmarkConfig,
        preprocessing: PreprocessingConfig | Any | None,
        model_params: dict[str, Any],
        search_space: Any | None,
        positive_class: Any | None,
        seed: int | None,
    ):
        final_plan = _protected_test_plan(plan, dataset)
        tuning = _with_seed(config.tuning, seed)
        optimization = self.tuning_engine.run(
            dataset=dataset,
            algorithm=algorithm,
            config=tuning,
            partition_plan=plan,
            preprocessing=preprocessing,
            search_space=search_space,
            evaluation_role="auto",
            require_complete=config.require_complete,
            model_params=model_params,
        )
        final_params = {**model_params, **optimization.best_params}
        validation = self.validation_engine.run(
            dataset=dataset,
            algorithm=algorithm,
            partition_plan=final_plan,
            preprocessing=preprocessing,
            metrics=config.metrics,
            evaluation_role="test",
            positive_class=positive_class,
            random_state=seed,
            return_estimators=config.return_estimators,
            require_complete=config.require_complete,
            **final_params,
        )
        validation.metadata.update(
            {
                "benchmark_tuned": True,
                "tuning_partition_fingerprint": plan.fingerprint,
                "tuning_refit_metric": optimization.refit_metric,
            }
        )
        return validation, optimization


def benchmark_models(registry: AlgorithmRegistry, **kwargs: Any) -> BenchmarkResult:
    """Functional convenience wrapper around :class:`BenchmarkEngine`."""

    return BenchmarkEngine(registry).run(**kwargs)


def _normalize_datasets(
    datasets: DatasetBundle | BenchmarkDataset | Mapping[str, DatasetBundle] | Sequence[BenchmarkDataset],
) -> tuple[BenchmarkDataset, ...]:
    if isinstance(datasets, BenchmarkDataset):
        result = (datasets,)
    elif isinstance(datasets, DatasetBundle):
        result = (BenchmarkDataset(label="dataset", dataset=datasets),)
    elif isinstance(datasets, Mapping):
        result = tuple(
            BenchmarkDataset(label=str(label), dataset=dataset, representation=str(label))
            for label, dataset in datasets.items()
        )
    else:
        result = tuple(datasets)

    if not result:
        raise BenchmarkContractError("At least one benchmark dataset is required.")
    if any(not isinstance(item, BenchmarkDataset) for item in result):
        raise BenchmarkContractError(
            "Sequence benchmark inputs must contain BenchmarkDataset objects."
        )
    labels = [item.label for item in result]
    if len(labels) != len(set(labels)):
        raise BenchmarkContractError("Benchmark dataset labels must be unique.")
    return result


def _normalize_partitions(
    partitions: PartitionPlan | BenchmarkPartition | Mapping[str, PartitionPlan] | Sequence[BenchmarkPartition],
) -> tuple[BenchmarkPartition, ...]:
    if isinstance(partitions, BenchmarkPartition):
        result = (partitions,)
    elif isinstance(partitions, PartitionPlan):
        result = (BenchmarkPartition(label="default", plan=partitions),)
    elif isinstance(partitions, Mapping):
        result = tuple(
            BenchmarkPartition(label=str(label), plan=plan)
            for label, plan in partitions.items()
        )
    else:
        result = tuple(partitions)
    if not result:
        raise BenchmarkContractError("At least one benchmark partition is required.")
    if any(not isinstance(item, BenchmarkPartition) for item in result):
        raise BenchmarkContractError(
            "Sequence benchmark partitions must contain BenchmarkPartition objects."
        )
    labels = [item.label for item in result]
    if len(labels) != len(set(labels)):
        raise BenchmarkContractError("Benchmark partition labels must be unique.")
    return result


def _validate_dataset_alignment(datasets: tuple[BenchmarkDataset, ...]) -> None:
    reference = datasets[0].dataset
    reference_targets = _targets_by_id(reference)
    reference_ids = set(reference_targets)
    for item in datasets[1:]:
        current_targets = _targets_by_id(item.dataset)
        if set(current_targets) != reference_ids:
            raise BenchmarkContractError(
                "Representation datasets must contain the same sample IDs."
            )
        for sample_id, target in reference_targets.items():
            if not _target_equal(target, current_targets[sample_id]):
                raise BenchmarkContractError(
                    "Representation datasets must contain identical targets for each sample ID."
                )


def _targets_by_id(dataset: DatasetBundle) -> dict[Any, Any]:
    return {
        sample_id: target.item() if isinstance(target, np.generic) else target
        for sample_id, target in zip(dataset.sample_ids, np.asarray(dataset.y), strict=True)
    }


def _target_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
    return bool(left == right)


def _infer_benchmark_task(registry: AlgorithmRegistry, algorithms: tuple[str, ...]) -> str | None:
    tasks = []
    for name in algorithms:
        try:
            task = registry.get(name).task
        except Exception:
            continue
        if task not in tasks:
            tasks.append(task)
    if len(tasks) > 1:
        raise BenchmarkContractError(
            f"A benchmark matrix cannot mix supervised tasks: {tasks!r}."
        )
    return tasks[0] if tasks else None


def _safe_get_spec(registry: AlgorithmRegistry, algorithm: str):
    try:
        return registry.get(algorithm)
    except Exception:
        return None


def _is_baseline(spec: Any | None, algorithm: str) -> bool:
    return bool(spec is not None and "baseline" in spec.tags) or algorithm.startswith("dummy_")


def _partition_reference(
    datasets: tuple[BenchmarkDataset, ...],
    label: str | None,
) -> BenchmarkDataset:
    if label is None:
        return datasets[0]
    for item in datasets:
        if item.label == label:
            return item
    raise BenchmarkContractError(
        f"partitioning_reference '{label}' is not a benchmark dataset label."
    )


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
        dataset_fingerprint=dataset.fingerprint,
        name=f"{split.name}_final_test",
        metadata={
            **dict(plan.metadata),
            "benchmark_final_test": True,
            "source_partition_fingerprint": plan.fingerprint,
        },
    )


def _with_seed(config: TuningConfig | None, seed: int | None) -> TuningConfig:
    if config is None:
        raise BenchmarkContractError("Tuned benchmark requires TuningConfig.")
    if config.random_state is not None:
        return config
    return replace(config, random_state=seed)


def _run_id(dataset: str, partition: str, algorithm: str, mode: str, seed: int | None) -> str:
    payload = "|".join((dataset, partition, algorithm, mode, repr(seed)))
    return sha256(payload.encode("utf-8")).hexdigest()[:16]
