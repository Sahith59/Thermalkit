#!/usr/bin/env python3
"""Phase 5 — Forge evaluation.

Runs inference through ForgeDaemon with bandit-chosen batch_size.
Records tok/sec, cpu_temp, chosen batch_size every call.

Usage:
    python benchmarks/eval_forge.py
    python benchmarks/eval_forge.py --rounds 30 --max-tokens 150
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from thermalkit.daemon.daemon import ForgeDaemon

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
    parser.add_argument("--fresh-policy", action="store_true",
                        help="Delete existing policy and start fresh.")
    parser.add_argument("--output", default="thermalkit",
                        help="Output filename stem (default: thermalkit → thermalkit.json)")
    args = parser.parse_args()

    policy_path = os.path.expanduser("~/.thermalkit/phase5_policy.npz")
    db_path = os.path.expanduser("~/.thermalkit/phase5_inference.db")

    if args.fresh_policy and os.path.exists(policy_path):
        os.remove(policy_path)
        print("Removed existing policy — starting fresh.", flush=True)

    print("=" * 65, flush=True)
    print("THERMALKIT PHASE 5 — FORGE EVALUATION", flush=True)
    print("=" * 65, flush=True)
    print(f"Model   : {args.model}", flush=True)
    print(f"Rounds  : {args.rounds}", flush=True)
    print(f"Tokens  : {args.max_tokens} per call", flush=True)
    print(f"Policy  : {policy_path}", flush=True)
    print("=" * 65, flush=True)

    print("\n[1/2] Loading model + starting daemon (please wait ~30s)...", flush=True)
    t0 = time.perf_counter()
    daemon = ForgeDaemon.from_model(
        args.model,
        policy_path=policy_path,
        db_path=db_path,
    )
    daemon.start()
    print(f"    Daemon ready in {time.perf_counter() - t0:.1f}s. "
          f"Policy has {daemon.call_count} prior calls.", flush=True)

    import time as _time
    _time.sleep(2)  # let sensor warm up
    s = daemon._sensor.get_current_state(timeout=1.0) or {}
    print(f"    Sensors — cpu={s.get('cpu_temp_c', 'n/a')}°C  "
          f"batt={s.get('batt_pct', 'n/a')}%", flush=True)

    print(f"\n[2/2] Running {args.rounds} Forge rounds...\n", flush=True)
    print(f"{'Rnd':<5} {'batch':<7} {'tok/s':<10} {'cpu°C':<9} {'reward':<13} {'wall'}", flush=True)
    print("-" * 57, flush=True)

    results = []
    tok_secs = []
    cpu_temps = []
    batch_choices = []

    import mlx_lm
    import mlx.core as mx

    for i in range(args.rounds):
        prompt = PROMPTS[i % len(PROMPTS)]

        # Capture what the daemon picks
        from thermalkit.mlx_wrap.state_builder import StateBuilder
        from thermalkit.bandit.action_space import batch_size_for
        sb = StateBuilder(daemon._sensor)
        state = sb.get()
        action_idx = daemon._policy.select_action(state)
        chosen_bs = batch_size_for(action_idx)

        t0 = time.perf_counter()
        daemon.generate(prompt, max_tokens=args.max_tokens)
        wall = time.perf_counter() - t0

        st = daemon.status()
        tok_sec = st["last_reward"] or 0.0
        cpu = st["cpu_temp_c"]

        # Get actual tok/sec from last meter entry
        meter_entries = list(daemon._meter._entries)
        actual_tps = meter_entries[-1][1] if meter_entries else 0.0

        tok_secs.append(actual_tps)
        if cpu:
            cpu_temps.append(cpu)
        batch_choices.append(chosen_bs)

        entry = {
            "round": i + 1,
            "ts": time.time(),
            "batch_size": chosen_bs,
            "tok_sec": round(actual_tps, 2),
            "cpu_temp_c": round(cpu, 2) if cpu else None,
            "wall_sec": round(wall, 3),
            "policy_calls": daemon.call_count,
            "reward": round(st["last_reward"], 2) if st["last_reward"] else None,
        }
        results.append(entry)

        cpu_str = f"{cpu:.1f}" if cpu else "n/a"
        reward_str = f"{st['last_reward']:.1f}" if st["last_reward"] else "pending"
        print(f"{i+1:<5} {chosen_bs:<7} {actual_tps:<10.1f} {cpu_str:<9} {reward_str:<13} {wall:.2f}s",
              flush=True)

    daemon.stop()

    # Summary stats
    import statistics
    tok_secs_clean = [t for t in tok_secs if t > 0]
    mean_tps = statistics.mean(tok_secs_clean) if tok_secs_clean else 0
    p5_tps = sorted(tok_secs_clean)[int(len(tok_secs_clean) * 0.05)] if tok_secs_clean else 0
    std_tps = statistics.stdev(tok_secs_clean) if len(tok_secs_clean) > 1 else 0
    max_cpu = max(cpu_temps) if cpu_temps else None
    mean_cpu = statistics.mean(cpu_temps) if cpu_temps else None
    throttle_events = sum(1 for t in cpu_temps if t > 92.0)

    from collections import Counter
    bs_counts = Counter(batch_choices)

    print("\n" + "=" * 65, flush=True)
    print("FORGE SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"  Mean tok/sec      : {mean_tps:.2f}", flush=True)
    print(f"  P5  tok/sec       : {p5_tps:.2f}  (throttle floor)", flush=True)
    print(f"  Std dev tok/sec   : {std_tps:.2f}  (stability)", flush=True)
    print(f"  Mean cpu_temp     : {mean_cpu:.2f}°C" if mean_cpu else "  Mean cpu_temp     : n/a", flush=True)
    print(f"  Max  cpu_temp     : {max_cpu:.2f}°C" if max_cpu else "  Max  cpu_temp     : n/a", flush=True)
    print(f"  Throttle events   : {throttle_events}  (temp > 92°C)", flush=True)
    print(f"  Policy calls      : {daemon.call_count}", flush=True)
    print(f"  Batch size dist   : {dict(sorted(bs_counts.items()))}", flush=True)

    summary = {
        "mode": "thermalkit",
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
        "policy_calls": daemon.call_count,
        "batch_size_distribution": dict(sorted(bs_counts.items())),
        "runs": results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{args.output}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
