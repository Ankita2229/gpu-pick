"""Instance ranking logic.

Filters candidates by VRAM and budget, ranks by throughput-per-dollar.
Prefers spot price when available; skips instances not available in the account.
"""

from __future__ import annotations

from dataclasses import dataclass
from .gpu_db import InstanceSpec, GPU_DB


@dataclass
class RankedInstance:
    rank: int
    instance: InstanceSpec
    on_demand_per_hr: float
    spot_per_hr: float | None       # None if spot unavailable
    effective_per_hr: float         # spot if available and --use-spot, else on-demand
    is_spot: bool
    vram_gb: float
    throughput_score: float
    cost_efficiency: float
    est_hours: float | None
    est_cost: float | None
    availability_note: str          # "available", "not offered in region", "quota=0 ..."


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
    use_spot: bool = True,
    min_gpus: int = 1,
) -> list[RankedInstance]:
    """Return top-n instances ranked by cost-efficiency, filtered by VRAM, budget,
    and actual availability. Unavailable instances are skipped entirely so the
    returned list always contains actionable options.
    """
    try:
        from .providers.aws import get_all_instances
        raw = get_all_instances(region)
        # raw: (sm_name, gpu_name, n_gpus, on_demand, spot|None, is_avail, note, source)
    except Exception:
        from .gpu_db import AWS_SAGEMAKER_INSTANCES
        raw = [
            (i.name, i.gpu_name, i.n_gpus, i.on_demand_per_hr, None, True, "available", "static")
            for i in AWS_SAGEMAKER_INSTANCES
        ]

    ranked = []
    for name, gpu_name, n_gpus, on_demand, spot, is_avail, note, source in raw:
        if not is_avail and note == "not offered in region":
            continue  # instance hardware genuinely not available here
        if n_gpus < min_gpus:
            continue
        if gpu_name not in GPU_DB:
            continue

        inst = _build_instance_spec(name, gpu_name, n_gpus, on_demand)

        if use_spot and spot is not None:
            effective_per_hr = spot
            is_spot_flag = True
        else:
            effective_per_hr = on_demand
            is_spot_flag = False

        if inst.total_vram_gb < required_vram_gb:
            continue
        if effective_per_hr > budget_per_hr:
            # On-demand over budget but maybe spot fits?
            if not (use_spot and spot is not None and spot <= budget_per_hr):
                continue
            effective_per_hr = spot
            is_spot_flag = True

        throughput = inst.throughput_score
        efficiency = throughput / effective_per_hr if effective_per_hr > 0 else 0

        ranked.append(RankedInstance(
            rank=0,
            instance=inst,
            on_demand_per_hr=on_demand,
            spot_per_hr=spot,
            effective_per_hr=effective_per_hr,
            is_spot=is_spot_flag,
            vram_gb=inst.total_vram_gb,
            throughput_score=throughput,
            cost_efficiency=efficiency,
            est_hours=None,
            est_cost=None,
            availability_note=note,
        ))

    ranked.sort(key=lambda r: (-r.cost_efficiency, r.effective_per_hr))
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked[:n_results]
