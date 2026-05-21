"""AWS SageMaker instance pricing provider.

Tries to fetch live pricing via boto3 AWS Pricing API.
Falls back to the static table in gpu_db.py if credentials are unavailable
or the pricing API call fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class PriceResult:
    instance_name: str
    on_demand_per_hr: float
    spot_per_hr: float | None   # None if spot not available for SageMaker
    is_spot_available: bool
    region: str
    source: str  # "live" or "static"


def get_live_sagemaker_prices(region: str = "us-west-2") -> dict[str, float]:
    """Fetch SageMaker training instance on-demand prices from AWS Pricing API.

    Returns dict of instance_name -> $/hr. Falls back to empty dict on failure.
    """
    try:
        import boto3
        client = boto3.client("pricing", region_name="us-east-1")  # pricing API only in us-east-1

        prices = {}
        paginator = client.get_paginator("get_products")
        pages = paginator.paginate(
            ServiceCode="AmazonSageMaker",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "component", "Value": "Hosting"},
            ],
        )
        for page in pages:
            for price_str in page["PriceList"]:
                item = json.loads(price_str)
                attrs = item.get("product", {}).get("attributes", {})
                instance = attrs.get("instanceName", "")
                if not instance:
                    continue
                terms = item.get("terms", {}).get("OnDemand", {})
                for term in terms.values():
                    for dim in term.get("priceDimensions", {}).values():
                        usd = float(dim.get("pricePerUnit", {}).get("USD", 0))
                        if usd > 0:
                            prices[instance] = usd
        return prices
    except Exception:
        return {}


def get_prices(instance_names: list[str], region: str = "us-west-2") -> dict[str, PriceResult]:
    """Return pricing for requested instances, live where possible."""
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
            spot_per_hr=None,       # SageMaker managed spot requires separate API
            is_spot_available=False,
            region=region,
            source=source,
        )
    return results
