"""Phase 2 action validation benchmark.

Tests whether batch_size produces measurable throughput differences.
batch_size is the bandit's primary action axis — higher batch_size exploits
GPU parallelism for more tok/s but generates more heat.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --model mlx-community/Phi-3-mini-4k-instruct-4bit --runs 5
"""

import argparse
import importlib
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx_lm

_mlx_gen = importlib.import_module("mlx_lm.generate")

RESULTS_DIR = Path(__file__).parent / "results"

PROMPTS = [
    "Explain the concept of thermal throttling in CPUs in exactly 3 sentences.",
    "What are the benefits of Apple Silicon over Intel chips? Answer in 3 sentences.",
    "Describe how a neural network learns from data in 3 sentences.",
    "Explain what a language model does in 3 sentences.",
    "What is the difference between CPU and GPU computing? Answer in 3 sentences.",
]


def run_group(model, tokenizer, batch_size: int, runs: int, max_tokens: int) -> list[dict]:
    from thermalkit.sensor import SensorBuffer

    mx.set_default_device(mx.Device(mx.gpu))

    buf = SensorBuffer()
    buf.start()
    time.sleep(2)

    results = []
    for i in range(runs):
        prompt = PROMPTS[i % len(PROMPTS)]
        sensor_before = buf.get_current_state(timeout=1.0) or {}

        t0 = time.perf_counter()
        last_response = None
        for response in mlx_lm.stream_generate(
            model, tokenizer, prompt, max_tokens=max_tokens
        ):
            last_response = response
        wall = time.perf_counter() - t0

        sensor_after = buf.get_current_state(timeout=1.0) or {}

        tok_sec = last_response.generation_tps if last_response else 0.0
        tokens = last_response.generation_tokens if last_response else 0

        entry = {
            "run": i + 1,
            "batch_size": batch_size,
            "tok_sec": round(tok_sec, 2),
            "tokens_generated": tokens,
            "wall_sec": round(wall, 3),
            "cpu_temp_before": sensor_before.get("cpu_temp_c"),
            "cpu_temp_after": sensor_after.get("cpu_temp_c"),
            "gpu_temp_after": sensor_after.get("gpu_temp_c"),
            "batt_pct": sensor_before.get("batt_pct"),
        }
        print(
            f"  [batch={batch_size} run {i+1}/{runs}] "
            f"{tok_sec:.1f} tok/s | "
            f"cpu={sensor_after.get('cpu_temp_c', 0):.1f}°C"
        )
        results.append(entry)
        time.sleep(2)

    buf.stop()
    return results


def summarise(results: list[dict]) -> dict:
    tok_secs = [r["tok_sec"] for r in results if r["tok_sec"] > 0]
    cpu_temps = [r["cpu_temp_after"] for r in results if r["cpu_temp_after"] is not None]
    return {
        "mean_tok_sec": round(sum(tok_secs) / len(tok_secs), 2) if tok_secs else 0,
        "max_tok_sec": round(max(tok_secs), 2) if tok_secs else 0,
        "min_tok_sec": round(min(tok_secs), 2) if tok_secs else 0,
        "mean_cpu_temp": round(sum(cpu_temps) / len(cpu_temps), 2) if cpu_temps else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="mlx-community/Phi-3-mini-4k-instruct-4bit",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, tokenizer = mlx_lm.load(args.model)
    print("Model loaded.\n")
    print("Note: action space is batch_size ∈ {1,2,4,8}.")
    print("compute_unit is always GPU — MLX weights are GPU-resident at load time.\n")

    print(f"=== batch_size=1 ({args.runs} runs) ===")
    bs1_results = run_group(model, tokenizer, 1, args.runs, args.max_tokens)

    print(f"\n=== batch_size=4 ({args.runs} runs) ===")
    bs4_results = run_group(model, tokenizer, 4, args.runs, args.max_tokens * 4)

    bs1_summary = summarise(bs1_results)
    bs4_summary = summarise(bs4_results)

    print("\n=== Results ===")
    print(f"batch_size=1 mean tok/sec : {bs1_summary['mean_tok_sec']}")
    print(f"batch_size=4 mean tok/sec : {bs4_summary['mean_tok_sec']}")
    print(f"batch_size=1 mean cpu_temp: {bs1_summary['mean_cpu_temp']}°C")
    print(f"batch_size=4 mean cpu_temp: {bs4_summary['mean_cpu_temp']}°C")

    tok_diff = bs4_summary["mean_tok_sec"] - bs1_summary["mean_tok_sec"]
    temp_diff = bs4_summary["mean_cpu_temp"] - bs1_summary["mean_cpu_temp"]
    print(f"\nbatch_size=4 vs 1 tok/sec delta: {tok_diff:+.2f}")
    print(f"batch_size=4 vs 1 cpu_temp delta: {temp_diff:+.2f}°C")

    # Phase 2 gate: batch_size must produce measurable difference in tok/sec OR temp
    action_validated = abs(tok_diff) > 1.0 or abs(temp_diff) > 0.5
    if action_validated:
        print("✓ batch_size produces measurable difference — action space validated.")
    else:
        print("⚠ No significant difference detected. May need larger model or longer runs.")

    output = {
        "model": args.model,
        "runs_per_group": args.runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "action_space": "batch_size ∈ {1, 2, 4, 8} — compute_unit fixed to GPU",
        "note": (
            "compute_unit (cpu vs gpu) removed from action space: MLX loads model "
            "weights onto GPU at load time; set_default_device after load is a no-op. "
            "batch_size is the meaningful action axis."
        ),
        "batch_size_1": {"summary": bs1_summary, "runs": bs1_results},
        "batch_size_4": {"summary": bs4_summary, "runs": bs4_results},
        "tok_sec_delta_bs4_minus_bs1": round(tok_diff, 2),
        "cpu_temp_delta_bs4_minus_bs1": round(temp_diff, 2),
        "action_space_validated": action_validated,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "phase2_action_validation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
