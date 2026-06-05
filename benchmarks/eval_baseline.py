#!/usr/bin/env python3
"""Phase 5 — Baseline evaluation.

Runs raw mlx_lm.generate() with no thermal awareness.
Records tok/sec and cpu_temp every call into a JSON file.

Usage:
    python benchmarks/eval_baseline.py
    python benchmarks/eval_baseline.py --rounds 30 --max-tokens 150
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx_lm

from thermalkit.sensor import SensorBuffer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PROMPTS = [
    "Explain what thermal throttling is and why it happens in modern processors.",
    "Describe how Apple Silicon differs from traditional x86 architectures.",
    "What is a contextual bandit algorithm and how does it differ from reinforcement learning?",
    "Explain the concept of unified memory in Apple Silicon chips.",
    "What is the difference between tok/sec burst performance and sustained performance?",
    "How does Metal GPU compute work on Apple Silicon?",
    "Describe the role of the Neural Engine in Apple Silicon inference.",
    "What causes temperature variance in sustained machine learning workloads?",
    "Explain how online learning algorithms adapt to changing environments.",
    "What is the UCB (Upper Confidence Bound) exploration strategy?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    print("=" * 65, flush=True)
    print("THERMALKIT PHASE 5 — BASELINE EVALUATION", flush=True)
    print("=" * 65, flush=True)
    print(f"Model   : {args.model}", flush=True)
    print(f"Rounds  : {args.rounds}", flush=True)
    print(f"Tokens  : {args.max_tokens} per call", flush=True)
    print("=" * 65, flush=True)

    print("\n[1/3] Loading model (please wait ~30s)...", flush=True)
    t0 = time.perf_counter()
    mx.set_default_device(mx.Device(mx.gpu))
    model, tokenizer = mlx_lm.load(args.model)
    print(f"    Model loaded in {time.perf_counter() - t0:.1f}s.", flush=True)

    print("\n[2/3] Starting sensors...", flush=True)
    buf = SensorBuffer()
    buf.start()
    time.sleep(2)
    s = buf.get_current_state(timeout=1.0) or {}
    print(f"    Sensors OK — cpu={s.get('cpu_temp_c', '?')}°C  "
          f"batt={s.get('batt_pct', '?')}%", flush=True)

    print(f"\n[3/3] Running {args.rounds} baseline rounds...\n", flush=True)
    print(f"{'Rnd':<5} {'tok/s':<10} {'tokens':<8} {'cpu°C':<9} {'wall'}", flush=True)
    print("-" * 45, flush=True)

    results = []
    tok_secs = []
    cpu_temps = []

    for i in range(args.rounds):
        prompt = PROMPTS[i % len(PROMPTS)]
        sensor_before = buf.get_current_state(timeout=0.5) or {}

        t0 = time.perf_counter()
        last_resp = None
        for resp in mlx_lm.stream_generate(model, tokenizer, prompt,
                                            max_tokens=args.max_tokens):
            last_resp = resp
        wall = time.perf_counter() - t0

        sensor_after = buf.get_current_state(timeout=0.5) or {}

        tok_sec = last_resp.generation_tps if last_resp else 0.0
        tokens = last_resp.generation_tokens if last_resp else 0
        cpu = sensor_after.get("cpu_temp_c") or sensor_before.get("cpu_temp_c")

        tok_secs.append(tok_sec)
        if cpu:
            cpu_temps.append(cpu)

        entry = {
            "round": i + 1,
            "ts": time.time(),
            "tok_sec": round(tok_sec, 2),
            "tokens_generated": tokens,
            "cpu_temp_c": round(cpu, 2) if cpu else None,
            "wall_sec": round(wall, 3),
            "start_cpu_temp_c": round(sensor_before.get("cpu_temp_c", 0), 2) if sensor_before.get("cpu_temp_c") else None,
        }
        results.append(entry)

        cpu_str = f"{cpu:.1f}" if cpu else "n/a"
        print(f"{i+1:<5} {tok_sec:<10.1f} {tokens:<8} {cpu_str:<9} {wall:.2f}s",
              flush=True)

    buf.stop()

    # Summary stats
    import statistics
    mean_tps = statistics.mean(tok_secs)
    p5_tps = sorted(tok_secs)[int(len(tok_secs) * 0.05)]
    std_tps = statistics.stdev(tok_secs) if len(tok_secs) > 1 else 0.0
    max_cpu = max(cpu_temps) if cpu_temps else None
    mean_cpu = statistics.mean(cpu_temps) if cpu_temps else None
    throttle_events = sum(1 for t in cpu_temps if t > 92.0)

    summary = {
        "mode": "baseline",
        "model": args.model,
        "rounds": args.rounds,
        "max_tokens": args.max_tokens,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mean_tok_sec": round(mean_tps, 2),
        "p5_tok_sec": round(p5_tps, 2),
        "std_tok_sec": round(std_tps, 2),
        "max_cpu_temp": round(max_cpu, 2) if max_cpu else None,
        "mean_cpu_temp": round(mean_cpu, 2) if mean_cpu else None,
        "throttle_events_above_92c": throttle_events,
        "runs": results,
    }

    print("\n" + "=" * 65, flush=True)
    print("BASELINE SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"  Mean tok/sec      : {mean_tps:.2f}", flush=True)
    print(f"  P5  tok/sec       : {p5_tps:.2f}  (throttle floor)", flush=True)
    print(f"  Std dev tok/sec   : {std_tps:.2f}  (stability)", flush=True)
    print(f"  Mean cpu_temp     : {mean_cpu:.2f}°C" if mean_cpu else "  Mean cpu_temp     : n/a", flush=True)
    print(f"  Max  cpu_temp     : {max_cpu:.2f}°C" if max_cpu else "  Max  cpu_temp     : n/a", flush=True)
    print(f"  Throttle events   : {throttle_events}  (temp > 92°C)", flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "baseline.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
