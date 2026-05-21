"""gpu-pick CLI — find the fastest, cheapest GPU instance for your ML run."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click


def _parse_yaml_workload(yaml_path: str) -> dict:
    """Extract workload params from an experiment YAML."""
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        raise click.ClickException(f"Could not read YAML: {e}")

    model_cfg = cfg.get("model", {})
    base_model = model_cfg.get("base", "")

    model_size = "7b"
    for size in ["405b", "72b", "70b", "34b", "32b", "30b", "14b", "13b", "8b", "7b", "3b", "1b"]:
        if size in base_model.lower():
            model_size = size
            break

    # Try to count training episodes from referenced data files.
    # Sources may live under trajectory.sources, data.sources, or a top-level
    # trajectory.path (the built trajectory file itself).
    n_episodes = None
    yaml_dir = Path(yaml_path).parent

    def _count_jsonl(path_str: str) -> int | None:
        if Path(path_str).is_absolute():
            candidates = [Path(path_str)]
        else:
            # Try yaml_dir and up to 3 parent directories (paths in YAMLs are
            # often relative to the project root, not the experiments subfolder)
            candidates = [yaml_dir / path_str]
            for parent in list(yaml_dir.parents)[:3]:
                candidates.append(parent / path_str)
        for p in candidates:
            if p.exists() and p.suffix in (".jsonl", ".json"):
                try:
                    return sum(1 for line in p.open() if line.strip())
                except Exception:
                    pass
        return None

    # Check trajectory.sources first (raw source episodes before grouping)
    traj_cfg = cfg.get("trajectory", {})
    sources = traj_cfg.get("sources", cfg.get("data", {}).get("sources", []))
    for src in sources:
        src_path = src.get("path", "") if isinstance(src, dict) else str(src)
        if src_path:
            cnt = _count_jsonl(src_path)
            if cnt is not None:
                n_episodes = (n_episodes or 0) + cnt

    # Fall back to counting the built trajectory file
    if n_episodes is None:
        traj_path = traj_cfg.get("path", "")
        if traj_path:
            n_episodes = _count_jsonl(traj_path)

    return {
        "model_size": model_size,
        "seq_len": int(model_cfg.get("max_seq_length", 2048)),
        "batch_size": int(model_cfg.get("batch_size", 2)),
        "grad_accum": int(model_cfg.get("grad_accum", 8)),
        "epochs": int(model_cfg.get("epochs", 2)),
        "lora_rank": int(model_cfg.get("lora_rank", 16)),
        "n_train_episodes": n_episodes,
        "name": cfg.get("name", Path(yaml_path).stem),
    }


def _estimate_steps(n_episodes: int | None, batch_size: int, n_gpus: int,
                    grad_accum: int, epochs: int, val_fraction: float = 0.2) -> int | None:
    if n_episodes is None:
        return None
    train_eps = int(n_episodes * (1 - val_fraction))
    eff_batch = batch_size * n_gpus * grad_accum
    steps_per_epoch = math.ceil(train_eps / eff_batch)
    return steps_per_epoch * epochs


def _steps_per_hr(throughput_score: float, seq_len: int) -> float:
    """Estimate gradient steps/hr for a given GPU throughput and sequence length.

    Baseline: A100 40GB (throughput=1.0) does ~60 steps/hr at seq_len=2048,
    batch=1, 7B model with flash attention. Step time scales linearly with
    seq_len (FFN dominates; flash-attn makes attention O(seq_len) in practice).
    """
    baseline_steps_per_hr = 60.0
    seq_scale = 2048 / max(seq_len, 1)
    return baseline_steps_per_hr * throughput_score * seq_scale


def _render_table(ranked, mode: str, vram_req: float, budget: float,
                  price_source: str) -> str:
    if not ranked:
        return f"  ✗ No {mode} instances fit VRAM={vram_req:.0f}GB and budget=${budget}/hr"

    lines = [f"\n{mode.upper()} — top {len(ranked)} (budget ${budget}/hr | VRAM req ~{vram_req:.0f} GB)"]
    lines.append(f"  Prices: {price_source} | Throughput: relative estimates (A100 40GB = 1.0)")
    lines.append("")

    show_time = any(r.est_hours is not None for r in ranked)
    header = (
        f"  {'#':>2}  {'Instance':<22} {'GPUs':<18} {'VRAM':>6}  "
        f"{'On-demand':>9}  {'Spot':>6}  {'Eff':>5}"
    )
    if show_time:
        header += f"  {'Est.hrs':>8}  {'Est.$':>8}"
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    needs_quota = []
    for r in ranked:
        gpu_str = f"{r.instance.n_gpus}×{r.instance.gpu_name}"
        spot_str = f"${r.spot_per_hr:.2f}" if r.spot_per_hr is not None else "   n/a"
        active_tag = "◀" if r.is_spot else " "
        quota_tag = "‡" if r.availability_note == "needs quota request" else " "
        tag = " ← best" if r.rank == 1 else ""
        row = (
            f"  {r.rank:>2}  {r.instance.name:<22} {gpu_str:<18} "
            f"{r.vram_gb:>5.0f}G  "
            f"${r.on_demand_per_hr:>7.2f}  {spot_str:>6}{active_tag} "
            f"{r.cost_efficiency:>5.2f}{quota_tag}"
        )
        if show_time:
            hrs = f"{r.est_hours:.1f}h" if r.est_hours is not None else "       ?"
            cost = f"${r.est_cost:.0f}" if r.est_cost is not None else "       ?"
            row += f"  {hrs:>8}  {cost:>8}"
        row += tag
        lines.append(row)
        if quota_tag == "‡":
            needs_quota.append(r.instance.name)

    footnotes = []
    if any(r.is_spot for r in ranked):
        footnotes.append("  ◀ spot price active — managed spot may interrupt; checkpoint frequently")
    if needs_quota:
        footnotes.append("  ‡ quota=0 in your account — request via AWS Service Quotas before use")
    if show_time:
        footnotes.append("  Est.hrs/Est.$ based on step-time model; actual time varies ±30%")
    if footnotes:
        lines.append("")
        lines.extend(footnotes)
    return "\n".join(lines)


@click.command()
@click.option("--yaml", "yaml_path", default=None, help="Path to experiment YAML")
@click.option("--model-size", default=None, help="Model size e.g. 7B, 32B, 70B")
@click.option("--seq-len", default=None, type=int, help="Max sequence length")
@click.option("--batch-size", default=None, type=int, help="Per-device batch size")
@click.option("--grad-accum", default=None, type=int, help="Gradient accumulation steps")
@click.option("--epochs", default=None, type=int, help="Number of training epochs")
@click.option("--n-episodes", default=None, type=int, help="Total training episodes (overrides YAML data count)")
@click.option("--budget", required=True, type=float, help="Max $/hr hard ceiling")
@click.option("--mode", default="both", type=click.Choice(["train", "eval", "both"]), help="Train, eval, or both")
@click.option("--provider", default="aws", type=click.Choice(["aws", "gcp", "all"]), help="Cloud provider")
@click.option("--region", default="us-west-2", help="Cloud region")
@click.option("--use-spot/--no-spot", default=True, help="Consider spot instances")
@click.option("--min-gpus", default=1, type=int, help="Minimum number of GPUs")
@click.option("--n-results", default=5, type=int, help="Number of results to show per mode")
@click.option("--lora-rank", default=16, type=int, help="LoRA rank (affects optimizer VRAM)")
@click.option("--json-output", is_flag=True, default=False, help="Output JSON instead of table")
def recommend(
    yaml_path, model_size, seq_len, batch_size, grad_accum, epochs,
    n_episodes, budget, mode, provider, region, use_spot, min_gpus,
    n_results, lora_rank, json_output,
):
    """Find the fastest, cheapest GPU instance for your ML run.

    \b
    Examples:
      gpu-pick recommend --yaml experiments/v4c.yaml --budget 50
      gpu-pick recommend --model-size 7B --seq-len 8192 --budget 50
      gpu-pick recommend --model-size 32B --budget 100 --mode train
    """
    from gpupick.memory import estimate_vram
    from gpupick.ranker import rank_instances

    # Resolve workload params — YAML auto-populates, CLI flags override
    workload = {}
    if yaml_path:
        workload = _parse_yaml_workload(yaml_path)

    resolved_model_size = model_size or workload.get("model_size", "7b")
    resolved_seq_len    = seq_len    or workload.get("seq_len", 2048)
    resolved_batch      = batch_size or workload.get("batch_size", 2)
    resolved_accum      = grad_accum or workload.get("grad_accum", 8)
    resolved_epochs     = epochs     or workload.get("epochs", 2)
    resolved_lora_rank  = lora_rank  or workload.get("lora_rank", 16)
    resolved_episodes   = n_episodes or workload.get("n_train_episodes")
    job_name            = workload.get("name", "")

    # Estimate VRAM
    try:
        vram = estimate_vram(
            model_size=resolved_model_size,
            seq_len=resolved_seq_len,
            batch_size=resolved_batch,
            grad_accum=resolved_accum,
            lora=True,
            lora_rank=resolved_lora_rank,
        )
    except ValueError as e:
        raise click.ClickException(str(e))

    # Header
    if not json_output:
        click.echo("")
        click.echo("gpu-pick — instance recommendation")
        click.echo("━" * 66)
        if job_name:
            click.echo(f"  Job:      {job_name}")
        click.echo(f"  Model:    {resolved_model_size.upper()} | seq_len={resolved_seq_len} | batch={resolved_batch} | grad_accum={resolved_accum}")
        click.echo(f"  VRAM req: {vram}")
        click.echo(f"  Budget:   ${budget}/hr hard ceiling")
        click.echo(f"  Provider: {provider.upper()} ({region})")
        if resolved_episodes:
            click.echo(f"  Episodes: {resolved_episodes:,} {'(from data file)' if not n_episodes else '(--n-episodes)'}")
        click.echo("━" * 66)

    results = {}

    # Get price source info
    price_source = "static fallback"
    try:
        from gpupick.providers.aws import get_live_sagemaker_prices
        live = get_live_sagemaker_prices(region)
        price_source = "live AWS Training pricing + EC2 spot" if live else "static fallback"
    except Exception:
        pass

    # Training recommendation
    if mode in ("train", "both"):
        train_ranked = rank_instances(
            required_vram_gb=vram.total_train_gb,
            budget_per_hr=budget,
            provider=provider,
            region=region,
            n_results=n_results,
            use_spot=use_spot,
            min_gpus=min_gpus,
        )
        for r in train_ranked:
            steps = _estimate_steps(
                resolved_episodes, resolved_batch,
                r.instance.n_gpus, resolved_accum, resolved_epochs,
            )
            if steps and r.throughput_score > 0:
                sph = _steps_per_hr(r.throughput_score, resolved_seq_len)
                r.est_hours = steps / sph
                r.est_cost = r.est_hours * r.effective_per_hr

        results["train"] = train_ranked
        if not json_output:
            click.echo(_render_table(train_ranked, "training", vram.total_train_gb,
                                     budget, price_source))

    # Eval recommendation
    if mode in ("eval", "both"):
        eval_ranked = rank_instances(
            required_vram_gb=vram.total_eval_gb,
            budget_per_hr=budget,
            provider=provider,
            region=region,
            n_results=n_results,
            use_spot=use_spot,
            min_gpus=1,
        )
        results["eval"] = eval_ranked
        if not json_output:
            click.echo(_render_table(eval_ranked, "eval", vram.total_eval_gb,
                                     budget, price_source))

    # Exit 1 if nothing found
    all_ranked = [r for v in results.values() for r in v]
    if not all_ranked:
        click.echo(
            "\n✗ No instances found within budget and VRAM constraints.\n"
            "  Try: increase --budget, reduce --seq-len, or use a smaller model.",
            err=True,
        )
        sys.exit(1)

    if not json_output:
        click.echo("\n  All throughput figures are estimates. Verify before committing to a run.")
        click.echo("")
    else:
        out = {
            m: [
                {
                    "rank": r.rank,
                    "instance": r.instance.name,
                    "gpu": f"{r.instance.n_gpus}×{r.instance.gpu_name}",
                    "vram_gb": r.vram_gb,
                    "on_demand_per_hr": r.on_demand_per_hr,
                    "spot_per_hr": r.spot_per_hr,
                    "effective_per_hr": r.effective_per_hr,
                    "is_spot": r.is_spot,
                    "efficiency": round(r.cost_efficiency, 3),
                    "availability": r.availability_note,
                    "est_hours": round(r.est_hours, 2) if r.est_hours is not None else None,
                    "est_cost": round(r.est_cost, 2) if r.est_cost is not None else None,
                }
                for r in ranked
            ]
            for m, ranked in results.items()
        }
        click.echo(json.dumps(out, indent=2))


@click.group()
def cli():
    """gpu-pick — find the fastest, cheapest GPU instance for your ML run."""
    pass


cli.add_command(recommend)


def main():
    cli()
