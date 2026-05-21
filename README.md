# gpu-pick

Find the fastest, cheapest GPU instance for your ML training and eval runs — before you commit to a job.

```
gpu-pick recommend --yaml experiments/v4c.yaml --budget 50
gpu-pick recommend --model-size 7B --seq-len 8192 --budget 50
gpu-pick recommend --model-size 32B --budget 100 --mode train
```

## Install

```bash
pip install -e .
```

Requires AWS credentials configured (`~/.aws/credentials` or env vars) for live pricing.
Falls back to a static price table if credentials are unavailable.

## Usage

```
gpu-pick recommend [OPTIONS]

Options:
  --yaml PATH           Experiment YAML (auto-detects model size, seq_len, batch)
  --model-size TEXT     Model size e.g. 7B, 32B, 70B
  --seq-len INT         Max sequence length  [default: 2048]
  --batch-size INT      Per-device batch size  [default: 2]
  --grad-accum INT      Gradient accumulation steps  [default: 8]
  --epochs INT          Training epochs  [default: 2]
  --n-episodes INT      Training episodes (for run time estimate)
  --budget FLOAT        Max $/hr — hard ceiling, required
  --mode                train | eval | both  [default: both]
  --provider            aws | gcp | all  [default: aws]
  --region TEXT         Cloud region  [default: us-west-2]
  --use-spot/--no-spot  Consider spot instances  [default: true]
  --min-gpus INT        Minimum number of GPUs
  --n-results INT       Results to show per mode  [default: 5]
  --json-output         Output JSON instead of table
```

## Example output

```
gpu-pick — instance recommendation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Job:      offline-auc-v4c
  Model:    7B | seq_len=8192 | batch=2 | grad_accum=8
  VRAM req: ~22 GB train | ~16 GB eval
  Budget:   $50.0/hr hard ceiling
  Provider: AWS (us-west-2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TRAINING — top 5 (budget $50/hr | VRAM req ~22 GB)
  Prices: live AWS pricing | Throughput: relative estimates (A100 40GB = 1.0)

   #  Instance               GPUs                VRAM    $/hr   Eff
  ─────────────────────────────────────────────────────────────────
   1  ml.g5.12xlarge         4×A10G               96G    7.09  0.15  ← best
   2  ml.p3.8xlarge          4×V100 16GB          64G   14.69  0.11
   3  ml.g5.48xlarge         8×A10G              192G   20.36  0.11
   4  ml.p4d.24xlarge        8×A100 40GB         320G   32.77  0.10
   5  ml.p3.16xlarge         8×V100 16GB         128G   28.15  0.10

EVAL — top 5 (budget $50/hr | VRAM req ~16 GB)
   1  ml.g4dn.xlarge         1×T4                 16G    0.74  0.30  ← best
   2  ml.g5.2xlarge          1×A10G               24G    1.21  0.23
   3  ml.g5.xlarge           1×A10G               24G    1.41  0.20

  All throughput figures are estimates. Verify before committing to a run.
```

## Spec

See [SPEC.md](SPEC.md) for the full behavioral contract — budget enforcement rules,
VRAM estimation methodology, ranking algorithm, and provider extension interface.

## Extending to other clouds

Add a file `gpupick/providers/<name>.py` implementing:

```python
def get_prices(instance_names: list[str], region: str) -> dict[str, PriceResult]: ...
```

Then register it in `gpupick/ranker.py`.
