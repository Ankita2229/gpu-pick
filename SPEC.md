# gpu-pick — Behavioral Spec

## Purpose

`gpu-pick` is a CLI tool that, given a workload description and a cost threshold,
finds the fastest GPU instance that fits within budget. It must run before any
training or eval job is submitted and must never allow a job to proceed on an
instance that exceeds the stated budget.

---

## Core Invariants (non-negotiable)

1. **Budget is a hard ceiling.** No instance above `--budget` ($/hr) may appear
   in the recommendation or be returned as the top pick. If nothing fits, the
   tool exits with a non-zero code and a clear message — it never silently picks
   the cheapest over-budget option.

2. **VRAM must fit.** The recommended instance must have enough GPU memory to
   hold model weights + activations + optimizer states for the stated workload.
   An instance that would OOM is never recommended, even if it is under budget.

3. **Cheapest-fastest trade-off is explicit.** When multiple instances fit budget
   and VRAM, the tool ranks by throughput-per-dollar (estimated steps/hr per
   dollar). The ranking is shown, not just the winner, so the user can make an
   informed choice.

4. **Training and eval are sized separately.** Eval requires less memory (no
   optimizer states, smaller activations). The tool produces separate
   recommendations for train and eval unless `--mode` is specified.

5. **Estimates are labeled as estimates.** The tool never presents throughput
   numbers as exact benchmarks. All throughput figures derive from a static GPU
   performance database and are marked as relative estimates.

6. **At least 3 options are shown** (or all that fit, if fewer than 3 exist)
   so the user always has a choice.

7. **Spot vs on-demand is explicit.** If a spot price is used, it is labeled
   as spot with a note that it may not be available. On-demand is always shown
   as a fallback.

---

## Inputs

| Input | Source | Required |
|---|---|---|
| `--yaml` | Experiment YAML file | One of yaml or manual flags |
| `--model-size` | e.g. `7B`, `32B`, `70B` | One of yaml or manual flags |
| `--seq-len` | Max sequence length | No (default: 2048) |
| `--batch-size` | Per-device batch size | No (default: 2) |
| `--grad-accum` | Gradient accumulation steps | No (default: 8) |
| `--epochs` | Number of training epochs | No (default: 2) |
| `--budget` | Max $/hr, e.g. `50` | Yes |
| `--mode` | `train`, `eval`, or `both` | No (default: both) |
| `--provider` | `aws`, `gcp`, `all` | No (default: aws) |
| `--use-spot` | Consider spot instances | No (default: true) |
| `--min-gpus` | Minimum number of GPUs | No |

---

## VRAM Estimation

VRAM requirement is estimated as:

```
model_vram    = params × dtype_bytes  (bf16 → 2 bytes/param)
lora_vram     = params × lora_fraction × 2  (LoRA adds ~1-2% of model size)
optimizer     = model_vram × 2  (AdamW: 2× model size in fp32 master weights)
activations   = batch_size × seq_len × hidden_dim × n_layers × dtype_bytes × 2
overhead      = 2 GB  (framework, CUDA kernels, etc.)

train_vram    = model_vram + lora_vram + optimizer + activations + overhead
eval_vram     = model_vram + activations_inference + overhead
                (no optimizer; activations_inference ≈ 0.3× train activations)
```

Hidden dim and n_layers are looked up from a static model size table. If the
exact model is unknown, the tool uses conservative estimates for its size class.

Multi-GPU: if `world_size > 1`, model_vram and optimizer are divided across GPUs
(DDP replicates; tensor parallel shards — tool assumes DDP for SageMaker jobs).

---

## Throughput Model

Relative throughput is derived from a static GPU benchmark database:

- Base unit: A100 40GB = 1.0
- All other GPUs expressed as a fraction/multiple of A100 40GB for LLM training
- Sources: MLPerf Training, published vendor benchmarks, community measurements
- Multi-GPU scaling: assumes 0.85 linear efficiency per additional GPU (conservative)

`throughput_score = gpu_tflops_relative × n_gpus × 0.85^(n_gpus-1)`
`cost_efficiency  = throughput_score / cost_per_hour`

---

## Ranking Algorithm

1. Filter: remove all instances where `cost_per_hour > budget`
2. Filter: remove all instances where `vram_per_gpu × n_gpus < required_vram`
3. Score: `cost_efficiency = throughput_score / cost_per_hour`
4. Sort descending by `cost_efficiency`
5. Return top N (default 5), clearly labeled

Tie-breaking: if two instances are within 5% cost efficiency, prefer the one
with lower absolute cost (cheaper wins when speed is equivalent).

---

## Output Format

```
gpu-pick — instance recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Workload:  7B model | seq_len=8192 | batch=2 | grad_accum=8
VRAM req:  ~22 GB train | ~16 GB eval
Budget:    $50/hr
Provider:  AWS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAINING (top 5)
 #  Instance           GPUs        VRAM    $/hr   Eff   Est. run
 1  ml.g5.12xlarge   4×A10G 24GB  96GB   $16.32  1.00  ~2.1h ← RECOMMENDED
 2  ml.p3.8xlarge    4×V100 16GB  64GB   $14.69  0.81  ~2.6h
 3  ml.g5.48xlarge   8×A10G 24GB  192GB  $32.64  0.92  ~1.1h
 4  ml.p4d.24xlarge  8×A100 40GB  320GB  $32.77  1.04  ~1.0h
 5  ml.p3.16xlarge   8×V100 16GB  128GB  $28.15  0.78  ~1.3h

EVAL (top 5)
 #  Instance           GPUs        VRAM    $/hr   Eff
 1  ml.g5.2xlarge    1×A10G 24GB  24GB   $1.21   1.00 ← RECOMMENDED
 2  ml.g5.4xlarge    1×A10G 24GB  24GB   $2.03   0.60
 3  ml.g4dn.xlarge   1×T4   16GB  16GB   $0.74   0.78

⚠  Spot prices shown where available. On-demand fallback always listed.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Machine-readable output available via `--json`.

---

## Exit Codes

| Code | Meaning |
|---|---|
| 0 | Recommendation produced successfully |
| 1 | No instance fits both VRAM and budget constraints |
| 2 | Provider credentials not found or pricing unavailable |
| 3 | Invalid input (bad YAML, unknown model size, etc.) |

---

## Provider Contract

Each provider module MUST implement:

```python
def get_instances(filters: InstanceFilters) -> list[InstanceSpec]:
    """Return all instances meeting min_gpus and available in the region."""

def get_price(instance: InstanceSpec, use_spot: bool) -> PriceResult:
    """Return on_demand and spot price (spot may be None if unavailable)."""
```

New providers (GCP, Lambda Labs, RunPod) are added by implementing this
interface in `gpupick/providers/<name>.py` and registering in `PROVIDERS`.

---

## Out of Scope

- Provisioning or launching instances (gpu-pick only recommends)
- Cost forecasting beyond a single job run
- Network egress or storage costs
- Multi-node distributed training (future)
- Quota checking (future)
