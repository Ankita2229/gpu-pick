"""Instance ranking logic.

Filters candidates by VRAM and budget, ranks by throughput-per-dollar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from .gpu_db import InstanceSpec, AWS_SAGEMAKER_INSTANCES


@dataclass
class RankedInstance:
    rank: int
    instance: InstanceSpec
    on_demand_per_hr: float
    effective_per_hr: float      # spot if available, else on-demand
    is_spot: bool
    vram_gb: float
    throughput_score: float
    cost_efficiency: float       # throughput / $/hr — higher is better
    est_hours: float | None      # estimated run time (None if steps unknown)
    est_cost: float | None       # estimated total cost
    fits_vram: bool
    under_budget: bool


def rank_instances(
    required_vram_gb: float,
    budget_per_hr: float,
    provider: str = "aws",
    region: str = "us-west-2",
    n_results: int = 5,
    total_steps: int | None = None,  # for est. run time
    use_spot: bool = True,
    min_gpus: int = 1,
) -> list[RankedInstance]:
    """Return instances ranked by cost-efficiency, filtered by VRAM and budget."""

    # Get candidate instances
    if provider == "aws":
        candidates = [i for i in AWS_SAGEMAKER_INSTANCES if i.n_gpus >= min_gpus]
    else:
        candidates = [i for i in AWS_SAGEMAKER_INSTANCES if i.n_gpus >= min_gpus]

    # Fetch live prices where possible
    try:
        from .providers.aws import get_prices
        price_map = get_prices([c.name for c in candidates], region=region)
    except Exception:
        price_map = {}

    ranked = []
    for inst in candidates:
        price_info = price_map.get(inst.name)
        on_demand = price_info.on_demand_per_hr if price_info else inst.on_demand_per_hr
        spot = price_info.spot_per_hr if price_info else None

        effective_per_hr = spot if (use_spot and spot is not None) else on_demand
        is_spot = use_spot and spot is not None

        fits_vram = inst.total_vram_gb >= required_vram_gb
        under_budget = effective_per_hr <= budget_per_hr

        if not fits_vram or not under_budget:
            continue

        throughput = inst.throughput_score
        efficiency = throughput / effective_per_hr if effective_per_hr > 0 else 0

        # Estimate run time: steps × estimated_seconds_per_step / 3600
        # Rough: A100 40GB does ~1 step/sec for 7B at seq2048 batch128
        # Scale by throughput_score and sequence length factor
        est_hours = None
        est_cost = None
        if total_steps is not None:
            # Baseline: A100 = 60 steps/hr at eff_batch=128, seq=2048
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
            fits_vram=fits_vram,
            under_budget=under_budget,
        ))

    # Sort by cost_efficiency descending; tie-break by absolute cost (cheaper wins)
    ranked.sort(key=lambda r: (-r.cost_efficiency, r.effective_per_hr))

    # Assign ranks
    for i, r in enumerate(ranked):
        r.rank = i + 1

    return ranked[:n_results]
