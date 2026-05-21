"""Instance ranking logic.

Filters candidates by VRAM and budget, ranks by throughput-per-dollar.
"""

from __future__ import annotations

from dataclasses import dataclass
from .gpu_db import InstanceSpec, GPU_DB


@dataclass
class RankedInstance:
    rank: int
    instance: InstanceSpec
    on_demand_per_hr: float
    effective_per_hr: float
    is_spot: bool
    vram_gb: float
    throughput_score: float
    cost_efficiency: float
    est_hours: float | None
    est_cost: float | None
    fits_vram: bool
    under_budget: bool


def _build_instance_spec(name: str, gpu_name: str, n_gpus: int, price: float) -> InstanceSpec:
    return InstanceSpec(
        name=name,
        provider="aws",
        gpu_name=gpu_name,
        n_gpus=n_gpus,
        on_demand_per_hr=price,
        instance_type=name,
    )


def rank_instances(
    required_vram_gb: float,
    budget_per_hr: float,
    provider: str = "aws",
    region: str = "us-west-2",
    n_results: int = 5,
    total_steps: int | None = None,
    use_spot: bool = True,
    min_gpus: int = 1,
) -> list[RankedInstance]:
    """Return instances ranked by cost-efficiency, filtered by VRAM and budget."""

    # Build candidate list from live catalog (falls back to static if live unavailable)
    try:
        from .providers.aws import get_all_instances
        raw_candidates = get_all_instances(region)
    except Exception:
        from .gpu_db import AWS_SAGEMAKER_INSTANCES
        raw_candidates = [
            (i.name, i.gpu_name, i.n_gpus, i.on_demand_per_hr, "static")
            for i in AWS_SAGEMAKER_INSTANCES
        ]

    ranked = []
    for name, gpu_name, n_gpus, on_demand, source in raw_candidates:
        if n_gpus < min_gpus:
            continue
        if gpu_name not in GPU_DB:
            continue

        inst = _build_instance_spec(name, gpu_name, n_gpus, on_demand)
        effective_per_hr = on_demand  # SageMaker managed spot handled separately
        is_spot = False

        if inst.total_vram_gb < required_vram_gb:
            continue
        if effective_per_hr > budget_per_hr:
            continue

        throughput = inst.throughput_score
        efficiency = throughput / effective_per_hr if effective_per_hr > 0 else 0

        est_hours = None
        est_cost = None
        if total_steps is not None:
            base_steps_per_hr = 60.0
            steps_per_hr = base_steps_per_hr * throughput
            est_hours = total_steps / steps_per_hr if steps_per_hr > 0 else None
            est_cost = est_hours * effective_per_hr if est_hours else None

        ranked.append(RankedInstance(
            rank=0,
            instance=inst,
            on_demand_per_hr=on_demand,
            effective_per_hr=effective_per_hr,
            is_spot=is_spot,
            vram_gb=inst.total_vram_gb,
            throughput_score=throughput,
            cost_efficiency=efficiency,
            est_hours=est_hours,
            est_cost=est_cost,
            fits_vram=True,
            under_budget=True,
        ))

    ranked.sort(key=lambda r: (-r.cost_efficiency, r.effective_per_hr))
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked[:n_results]
