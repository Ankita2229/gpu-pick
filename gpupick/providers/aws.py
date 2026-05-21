"""AWS SageMaker instance pricing provider.

Fetches live training instance catalog + prices via the AWS Pricing API.
Falls back to the static table in gpu_db.py only if the API is unavailable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass
class PriceResult:
    instance_name: str
    on_demand_per_hr: float
    spot_per_hr: float | None
    is_spot_available: bool
    region: str
    source: str  # "live" or "static"


# Maps SageMaker instance family prefix → (gpu_name, n_gpus_per_instance)
# Kept here because GPU hardware doesn't change; only pricing and instance
# availability change. Add new families when AWS launches them.
_FAMILY_TO_GPU: dict[str, tuple[str, int]] = {
    "ml.p5.48xlarge":    ("H100 80GB", 8),
    "ml.p4de.24xlarge":  ("A100 80GB", 8),
    "ml.p4d.24xlarge":   ("A100 40GB", 8),
    "ml.g6e.48xlarge":   ("H100 80GB", 8),   # g6e = L40S in some regions, override below
    "ml.g6.48xlarge":    ("L4",        8),
    "ml.g6.12xlarge":    ("L4",        4),
    "ml.g6.4xlarge":     ("L4",        1),
    "ml.g6.2xlarge":     ("L4",        1),
    "ml.g6.xlarge":      ("L4",        1),
    "ml.g5.48xlarge":    ("A10G",      8),
    "ml.g5.24xlarge":    ("A10G",      4),
    "ml.g5.12xlarge":    ("A10G",      4),
    "ml.g5.8xlarge":     ("A10G",      1),
    "ml.g5.4xlarge":     ("A10G",      1),
    "ml.g5.2xlarge":     ("A10G",      1),
    "ml.g5.xlarge":      ("A10G",      1),
    "ml.g4dn.16xlarge":  ("T4",        1),
    "ml.g4dn.12xlarge":  ("T4",        4),
    "ml.g4dn.8xlarge":   ("T4",        1),
    "ml.g4dn.4xlarge":   ("T4",        1),
    "ml.g4dn.2xlarge":   ("T4",        1),
    "ml.g4dn.xlarge":    ("T4",        1),
    "ml.p3dn.24xlarge":  ("V100 32GB", 8),
    "ml.p3.16xlarge":    ("V100 16GB", 8),
    "ml.p3.8xlarge":     ("V100 16GB", 4),
    "ml.p3.2xlarge":     ("V100 16GB", 1),
}


def get_live_sagemaker_prices(region: str = "us-west-2") -> dict[str, float]:
    """Fetch all SageMaker *training* instance prices for the region.

    Returns dict of instance_name -> $/hr. Empty dict on failure.
    """
    try:
        import boto3
        client = boto3.client("pricing", region_name="us-east-1")
        prices: dict[str, float] = {}
        paginator = client.get_paginator("get_products")
        pages = paginator.paginate(
            ServiceCode="AmazonSageMaker",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode",  "Value": region},
                {"Type": "TERM_MATCH", "Field": "component",   "Value": "Training"},
            ],
        )
        for page in pages:
            for price_str in page["PriceList"]:
                item = json.loads(price_str)
                attrs = item.get("product", {}).get("attributes", {})
                instance = attrs.get("instanceName", "")
                if not instance:
                    continue
                for term in item.get("terms", {}).get("OnDemand", {}).values():
                    for dim in term.get("priceDimensions", {}).values():
                        usd = float(dim.get("pricePerUnit", {}).get("USD", 0))
                        if usd > 0:
                            prices[instance] = usd
        return prices
    except Exception:
        return {}


def get_all_instances(region: str = "us-west-2") -> list[tuple[str, str, int, float, str]]:
    """Return list of (instance_name, gpu_name, n_gpus, price_per_hr, source).

    Queries live Pricing API first; if it returns data, builds the full catalog
    from live results. Falls back to the static table only if live fails.
    """
    from gpupick.gpu_db import GPU_DB, AWS_SAGEMAKER_INSTANCES

    live_prices = get_live_sagemaker_prices(region)

    if live_prices:
        rows = []
        for name, price in live_prices.items():
            if name not in _FAMILY_TO_GPU:
                continue  # unknown GPU family — skip rather than guess wrong
            gpu_name, n_gpus = _FAMILY_TO_GPU[name]
            if gpu_name not in GPU_DB:
                continue  # GPU not in our spec DB
            rows.append((name, gpu_name, n_gpus, price, "live"))
        if rows:
            return rows

    # Full fallback: static table
    return [
        (inst.name, inst.gpu_name, inst.n_gpus, inst.on_demand_per_hr, "static")
        for inst in AWS_SAGEMAKER_INSTANCES
    ]


def get_prices(instance_names: list[str], region: str = "us-west-2") -> dict[str, PriceResult]:
    """Return pricing for requested instances."""
    from gpupick.gpu_db import AWS_SAGEMAKER_INSTANCES

    static_map = {inst.name: inst for inst in AWS_SAGEMAKER_INSTANCES}
    live_prices = get_live_sagemaker_prices(region)
    source = "live" if live_prices else "static"

    results = {}
    for name in instance_names:
        if name not in static_map:
            continue
        static = static_map[name]
        on_demand = live_prices.get(name, static.on_demand_per_hr)
        results[name] = PriceResult(
            instance_name=name,
            on_demand_per_hr=on_demand,
            spot_per_hr=None,
            is_spot_available=False,
            region=region,
            source=source,
        )
    return results
