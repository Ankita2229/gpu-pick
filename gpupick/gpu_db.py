"""Static GPU specs and relative throughput database.

Throughput is relative to A100 40GB = 1.0 for LLM training (bf16, large batch).
Sources: MLPerf Training v3.1, vendor datasheets, community benchmarks.
All figures are estimates — labeled as such in output.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str           # e.g. "A100 40GB"
    vram_gb: float      # per-GPU VRAM
    tflops_bf16: float  # BF16 tensor core TFLOPs
    throughput_rel: float  # relative to A100 40GB = 1.0
    memory_bw_tbps: float  # memory bandwidth TB/s


GPU_DB: dict[str, GPUSpec] = {
    "H100 80GB":   GPUSpec("H100 80GB",   80,  989.0, 3.20, 3.35),
    "H100 40GB":   GPUSpec("H100 40GB",   40,  756.0, 2.60, 2.00),
    "A100 80GB":   GPUSpec("A100 80GB",   80,  312.0, 1.40, 2.00),
    "A100 40GB":   GPUSpec("A100 40GB",   40,  312.0, 1.00, 1.56),
    "A10G":        GPUSpec("A10G",        24,   31.2, 0.28, 0.60),
    "V100 32GB":   GPUSpec("V100 32GB",   32,  112.0, 0.52, 0.90),
    "V100 16GB":   GPUSpec("V100 16GB",   16,  112.0, 0.48, 0.90),
    "T4":          GPUSpec("T4",          16,   65.0, 0.22, 0.30),
    "L4":          GPUSpec("L4",          24,  121.0, 0.40, 0.48),
    "L40S":        GPUSpec("L40S",        48,  366.0, 0.85, 0.86),
    "A40":         GPUSpec("A40",         48,  149.7, 0.45, 0.70),
}


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    provider: str       # "aws", "gcp"
    gpu_name: str
    n_gpus: int
    on_demand_per_hr: float
    instance_type: str  # raw cloud instance type

    @property
    def gpu(self) -> GPUSpec:
        return GPU_DB[self.gpu_name]

    @property
    def total_vram_gb(self) -> float:
        return self.gpu.vram_gb * self.n_gpus

    @property
    def throughput_score(self) -> float:
        """Multi-GPU throughput with 0.85 linear scaling efficiency per GPU."""
        single = self.gpu.throughput_rel
        scaling = sum(0.85 ** i for i in range(self.n_gpus))
        return single * scaling

    @property
    def cost_efficiency(self) -> float:
        """Throughput per dollar — higher is better."""
        if self.on_demand_per_hr <= 0:
            return 0.0
        return self.throughput_score / self.on_demand_per_hr


# Static AWS SageMaker instance table (on-demand, us-west-2, 2025)
# Prices from AWS pricing page — fetched live when possible, this is fallback.
AWS_SAGEMAKER_INSTANCES: list[InstanceSpec] = [
    # A100 40GB instances
    InstanceSpec("ml.p4d.24xlarge",  "aws", "A100 40GB", 8,  32.77, "ml.p4d.24xlarge"),
    # A100 80GB instances
    InstanceSpec("ml.p4de.24xlarge", "aws", "A100 80GB", 8,  40.97, "ml.p4de.24xlarge"),
    # H100 instances
    InstanceSpec("ml.p5.48xlarge",   "aws", "H100 80GB", 8, 98.32, "ml.p5.48xlarge"),
    # A10G instances
    InstanceSpec("ml.g5.xlarge",     "aws", "A10G",      1,  1.41,  "ml.g5.xlarge"),
    InstanceSpec("ml.g5.2xlarge",    "aws", "A10G",      1,  1.21,  "ml.g5.2xlarge"),
    InstanceSpec("ml.g5.4xlarge",    "aws", "A10G",      1,  2.03,  "ml.g5.4xlarge"),
    InstanceSpec("ml.g5.8xlarge",    "aws", "A10G",      1,  3.67,  "ml.g5.8xlarge"),
    InstanceSpec("ml.g5.12xlarge",   "aws", "A10G",      4,  7.09,  "ml.g5.12xlarge"),
    InstanceSpec("ml.g5.24xlarge",   "aws", "A10G",      4, 10.18,  "ml.g5.24xlarge"),
    InstanceSpec("ml.g5.48xlarge",   "aws", "A10G",      8, 20.36,  "ml.g5.48xlarge"),
    # V100 instances
    InstanceSpec("ml.p3.2xlarge",    "aws", "V100 16GB", 1,  4.23,  "ml.p3.2xlarge"),
    InstanceSpec("ml.p3.8xlarge",    "aws", "V100 16GB", 4, 14.69,  "ml.p3.8xlarge"),
    InstanceSpec("ml.p3.16xlarge",   "aws", "V100 16GB", 8, 28.15,  "ml.p3.16xlarge"),
    InstanceSpec("ml.p3dn.24xlarge", "aws", "V100 32GB", 8, 35.89,  "ml.p3dn.24xlarge"),
    # T4 instances
    InstanceSpec("ml.g4dn.xlarge",   "aws", "T4",        1,  0.74,  "ml.g4dn.xlarge"),
    InstanceSpec("ml.g4dn.2xlarge",  "aws", "T4",        1,  1.17,  "ml.g4dn.2xlarge"),
    InstanceSpec("ml.g4dn.4xlarge",  "aws", "T4",        1,  1.68,  "ml.g4dn.4xlarge"),
    InstanceSpec("ml.g4dn.8xlarge",  "aws", "T4",        1,  2.72,  "ml.g4dn.8xlarge"),
    InstanceSpec("ml.g4dn.12xlarge", "aws", "T4",        4,  5.44,  "ml.g4dn.12xlarge"),
    InstanceSpec("ml.g4dn.16xlarge", "aws", "T4",        1,  5.44,  "ml.g4dn.16xlarge"),
]


# Model size → (n_params_billions, hidden_dim, n_layers)
MODEL_ARCH: dict[str, tuple[float, int, int]] = {
    "1b":   (1,    2048, 22),
    "3b":   (3,    3200, 26),
    "7b":   (7,    4096, 32),
    "8b":   (8,    4096, 32),
    "13b":  (13,   5120, 40),
    "14b":  (14,   5120, 40),
    "30b":  (30,   6656, 60),
    "32b":  (32,   6656, 60),
    "34b":  (34,   7168, 48),
    "70b":  (70,   8192, 80),
    "72b":  (72,   8192, 80),
    "405b": (405, 16384, 126),
}
