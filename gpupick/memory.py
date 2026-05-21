"""VRAM requirement estimator for LLM training and inference.

All estimates are conservative (round up). Labeled as estimates in output.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from .gpu_db import MODEL_ARCH


@dataclass
class VRAMEstimate:
    model_gb: float
    optimizer_gb: float
    activations_gb: float
    overhead_gb: float
    total_train_gb: float
    total_eval_gb: float

    def __str__(self) -> str:
        return (
            f"~{self.total_train_gb:.0f} GB train | "
            f"~{self.total_eval_gb:.0f} GB eval"
        )


def _parse_model_size(model_size: str) -> tuple[float, int, int]:
    """Return (n_params_B, hidden_dim, n_layers) from e.g. '7B' or '7b'."""
    key = model_size.lower().replace("b", "") + "b"
    # Try exact match first
    if key in MODEL_ARCH:
        return MODEL_ARCH[key]
    # Try numeric lookup — find closest
    try:
        target = float(model_size.lower().rstrip("b"))
    except ValueError:
        raise ValueError(f"Unknown model size '{model_size}'. Use e.g. 7B, 32B, 70B.")
    closest = min(MODEL_ARCH.keys(), key=lambda k: abs(float(k.rstrip("b")) - target))
    return MODEL_ARCH[closest]


def estimate_vram(
    model_size: str,
    seq_len: int = 2048,
    batch_size: int = 2,
    grad_accum: int = 8,
    dtype_bytes: int = 2,       # bf16
    lora: bool = True,
    lora_rank: int = 16,
) -> VRAMEstimate:
    """Estimate VRAM needed for training and eval."""
    params_b, hidden_dim, n_layers = _parse_model_size(model_size)
    params = params_b * 1e9

    # Model weights in bf16
    model_gb = (params * dtype_bytes) / 1e9

    # LoRA adds ~lora_rank/hidden_dim fraction of model size (rough)
    lora_fraction = (lora_rank / hidden_dim) if lora else 0.0
    lora_gb = model_gb * lora_fraction * 2  # q,v projections both sides

    if lora:
        # LoRA training: optimizer states only for LoRA params
        lora_params = params * lora_fraction * 2
        optimizer_gb = (lora_params * 4 * 2) / 1e9  # fp32 momentum + variance
    else:
        # Full fine-tuning: AdamW stores fp32 master weights + momentum + variance
        optimizer_gb = (params * 4 * 3) / 1e9

    # Activation memory (per layer, per token, rough estimate)
    # Each layer stores: attention (seq²), FFN intermediates, layer norms
    # Conservative: hidden_dim * seq_len * batch * n_layers * dtype * 4 (factor for intermediates)
    activation_elements = hidden_dim * seq_len * batch_size * n_layers * 4
    activations_gb = (activation_elements * dtype_bytes) / 1e9

    # Gradient checkpointing reduces activations by ~sqrt(n_layers)
    activations_gb = activations_gb / math.sqrt(n_layers)

    overhead_gb = 2.0  # CUDA kernels, framework, misc

    total_train_gb = model_gb + lora_gb + optimizer_gb + activations_gb + overhead_gb
    # Eval: no optimizer, reduced activations (no grad), no LoRA training states
    eval_activations_gb = (activation_elements * dtype_bytes) / 1e9 * 0.3 / math.sqrt(n_layers)
    total_eval_gb = model_gb + eval_activations_gb + overhead_gb

    return VRAMEstimate(
        model_gb=round(model_gb, 1),
        optimizer_gb=round(optimizer_gb, 1),
        activations_gb=round(activations_gb, 1),
        overhead_gb=overhead_gb,
        total_train_gb=math.ceil(total_train_gb),
        total_eval_gb=math.ceil(total_eval_gb),
    )
