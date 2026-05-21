"""AWS SageMaker instance pricing provider.

Fetches live training instance catalog + prices via the AWS Pricing API.
Fetches EC2 spot prices as proxy for SageMaker managed spot savings.
Checks real instance availability in the account/region before recommending.
Falls back to the static table in gpu_db.py only if the API is unavailable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class InstanceAvailability:
    instance_name: str
    on_demand_per_hr: float
    spot_per_hr: float | None       # None = spot not available / unknown
    spot_savings_pct: float | None  # e.g. 70.0 means 70% cheaper than on-demand
    is_available: bool              # False = instance not offered in this region/account
    availability_note: str          # e.g. "available", "quota=0", "not offered in region"
    source: str                     # "live" or "static"


# SageMaker instance name → (gpu_name, n_gpus)
# GPU hardware specs don't change; only pricing and availability change.
FAMILY_TO_GPU: dict[str, tuple[str, int]] = {
    "ml.p5.48xlarge":    ("H100 80GB", 8),
    "ml.p4de.24xlarge":  ("A100 80GB", 8),
    "ml.p4d.24xlarge":   ("A100 40GB", 8),
    "ml.g6e.48xlarge":   ("L40S",      8),
    "ml.g6e.12xlarge":   ("L40S",      4),
    "ml.g6e.2xlarge":    ("L40S",      1),
    "ml.g6e.xlarge":     ("L40S",      1),
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


def _sm_to_ec2(sm_name: str) -> str:
    """Strip 'ml.' prefix to get EC2 instance type from SageMaker name."""
    return sm_name.removeprefix("ml.")


def get_live_sagemaker_prices(region: str = "us-west-2") -> dict[str, float]:
    """Fetch SageMaker Training on-demand prices. Returns {} on failure."""
    try:
        import boto3
        client = boto3.client("pricing", region_name="us-east-1")
        prices: dict[str, float] = {}
        paginator = client.get_paginator("get_products")
        for page in paginator.paginate(
            ServiceCode="AmazonSageMaker",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "component",  "Value": "Training"},
            ],
        ):
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


def get_spot_prices(sm_names: list[str], region: str = "us-west-2") -> dict[str, float]:
    """Fetch current EC2 spot prices as proxy for SageMaker managed spot.

    SageMaker managed spot runs on EC2 spot under the hood and charges at
    approximately the same rate. Returns dict of sm_name -> $/hr.
    """
    if not sm_names:
        return {}
    ec2_types = list({_sm_to_ec2(n) for n in sm_names})
    ec2_to_sm = {_sm_to_ec2(n): n for n in sm_names}
    try:
        import boto3
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_spot_price_history(
            InstanceTypes=ec2_types,
            ProductDescriptions=["Linux/UNIX"],
            MaxResults=len(ec2_types) * 5,
        )
        # Keep cheapest current price per instance type across AZs
        best: dict[str, float] = {}
        for item in resp.get("SpotPriceHistory", []):
            ec2_type = item["InstanceType"]
            price = float(item["SpotPrice"])
            if ec2_type not in best or price < best[ec2_type]:
                best[ec2_type] = price
        return {ec2_to_sm[t]: p for t, p in best.items() if t in ec2_to_sm}
    except Exception:
        return {}


def check_instance_availability(sm_names: list[str], region: str = "us-west-2") -> dict[str, str]:
    """Check which instances are actually offered in this region.

    Returns dict of sm_name -> note ("available", "not offered in region", "quota=0").
    Assumes available for all names on API failure (fail open).
    """
    if not sm_names:
        return {}

    ec2_types = list({_sm_to_ec2(n) for n in sm_names})
    ec2_to_sm: dict[str, list[str]] = {}
    for n in sm_names:
        ec2_to_sm.setdefault(_sm_to_ec2(n), []).append(n)

    result = {n: "available" for n in sm_names}

    try:
        import boto3

        # 1. Check EC2 instance type offerings in the region.
        # Only mark unavailable if the API returned data AND the type is absent.
        # An empty response likely means API issue — don't filter in that case.
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.describe_instance_type_offerings(
            LocationType="region",
            Filters=[{"Name": "instance-type", "Values": ec2_types}],
        )
        offered_ec2 = {o["InstanceType"] for o in resp.get("InstanceTypeOfferings", [])}
        if offered_ec2:  # only filter if we got a real response
            not_offered_ec2 = set(ec2_types) - offered_ec2
            for ec2_type in not_offered_ec2:
                for sm in ec2_to_sm.get(ec2_type, []):
                    result[sm] = "not offered in region"

        # 2. Check SageMaker service quotas for Training.
        # SageMaker defaults to quota=0 for all GPU instances — user must request
        # each type. We tag quota=0 as a warning but do NOT mark as unavailable,
        # since quota requests are routine and the instance itself exists.
        # Quota names look like "ml.p4d.24xlarge for training job usage"
        try:
            sq = boto3.client("service-quotas", region_name=region)
            paginator = sq.get_paginator("list_service_quotas")
            for page in paginator.paginate(ServiceCode="sagemaker"):
                for q in page.get("Quotas", []):
                    qname = q.get("QuotaName", "")
                    for sm_name in sm_names:
                        if sm_name in qname and "training" in qname.lower():
                            if q.get("Value", 1) == 0 and result[sm_name] == "available":
                                result[sm_name] = "needs quota request"
                            break
        except Exception:
            pass  # quotas unavailable — leave as available

    except Exception:
        pass  # fail open — caller gets "available" for all

    return result


def get_all_instances(
    region: str = "us-west-2",
) -> list[tuple[str, str, int, float, float | None, bool, str, str]]:
    """Return full instance catalog with live prices and availability.

    Each row: (sm_name, gpu_name, n_gpus, on_demand_$/hr, spot_$/hr|None,
               is_available, availability_note, source)
    """
    from gpupick.gpu_db import GPU_DB, AWS_SAGEMAKER_INSTANCES

    # --- On-demand prices ---
    live_prices = get_live_sagemaker_prices(region)
    source = "live" if live_prices else "static"

    if live_prices:
        candidates = []
        for name, price in live_prices.items():
            if name not in FAMILY_TO_GPU:
                continue
            gpu_name, n_gpus = FAMILY_TO_GPU[name]
            if gpu_name not in GPU_DB:
                continue
            candidates.append((name, gpu_name, n_gpus, price))
    else:
        candidates = [
            (i.name, i.gpu_name, i.n_gpus, i.on_demand_per_hr)
            for i in AWS_SAGEMAKER_INSTANCES
        ]

    if not candidates:
        return []

    sm_names = [c[0] for c in candidates]

    # --- Spot prices ---
    spot_map = get_spot_prices(sm_names, region)

    # --- Availability ---
    avail_map = check_instance_availability(sm_names, region)

    rows = []
    for name, gpu_name, n_gpus, on_demand in candidates:
        spot = spot_map.get(name)
        is_avail = avail_map.get(name, "available") == "available"
        note = avail_map.get(name, "available")
        rows.append((name, gpu_name, n_gpus, on_demand, spot, is_avail, note, source))

    return rows
