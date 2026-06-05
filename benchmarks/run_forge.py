#!/usr/bin/env python3
"""Standalone Forge benchmark — bypasses the CLI, runs directly with python."""
import sys
import time
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60, flush=True)
print("THERMALKIT BENCHMARK", flush=True)
print("=" * 60, flush=True)

print("\n[1/4] Importing MLX and Forge...", flush=True)
import mlx.core as mx
import mlx_lm
from thermalkit.sensor import SensorBuffer
from thermalkit.bandit.policy import LinUCBPolicy
from thermalkit.mlx_wrap.state_builder import StateBuilder
from thermalkit.mlx_wrap.tok_sec_meter import TokSecMeter
from thermalkit.mlx_wrap.inference_log import InferenceLog
from thermalkit.bandit.action_space import batch_size_for, N_ACTIONS, BATCH_SIZES
print("    Imports OK.", flush=True)

MODEL = "mlx-community/Phi-3-mini-4k-instruct-4bit"
ROUNDS = 10
MAX_TOKENS = 30

print(f"\n[2/4] Loading model: {MODEL}", flush=True)
print("    This takes 20-30s — please wait...", flush=True)
t_load = time.perf_counter()
model, tokenizer = mlx_lm.load(MODEL)
print(f"    Model loaded in {time.perf_counter() - t_load:.1f}s.", flush=True)

print("\n[3/4] Starting sensor buffer...", flush=True)
buf = SensorBuffer()
buf.start()
time.sleep(2)
sensor_check = buf.get_current_state(timeout=1.0)
print(f"    Sensor OK — cpu_temp={sensor_check.get('cpu_temp_c', '?')}°C", flush=True)

print(f"\n[4/4] Running {ROUNDS} rounds of bandit-driven inference...\n", flush=True)

policy = LinUCBPolicy(alpha=1.0)
meter = TokSecMeter()
log = InferenceLog()
log.start()

PROMPTS = [
    "Name a color.", "What is 2+2?", "Say hi.", "Name a planet.",
    "What is Python?", "What is gravity?", "Name a fruit.",
    "What is the sun?", "Say goodbye.", "Name a country.",
]

last_state = None
last_action = None
last_tok_sec = None
results = []

print(f"{'Rnd':<5} {'batch':<7} {'tok/s':<10} {'cpu°C':<9} {'reward':<12} {'wall'}", flush=True)
print("-" * 55, flush=True)

for i in range(ROUNDS):
    # Build state
    sb = StateBuilder(buf)
    state = sb.get()
    sensor = buf.get_current_state(timeout=0.5) or {}

    # Update policy from previous call
    if last_state is not None and last_tok_sec is not None:
        reward = meter.get_reward() or last_tok_sec
        policy.update(last_state, last_action, reward)

    # Bandit picks action
    action_idx = policy.select_action(state)
    batch_size = batch_size_for(action_idx)

    # Inference
    mx.set_default_device(mx.Device(mx.gpu))
    t0 = time.perf_counter()
    last_resp = None
    for resp in mlx_lm.stream_generate(model, tokenizer, PROMPTS[i], max_tokens=MAX_TOKENS):
        last_resp = resp
    wall = time.perf_counter() - t0

    tok_sec = last_resp.generation_tps if last_resp else 0.0
    meter.record_tps(tok_sec)

    # Log
    log.log({
        "ts": time.time(),
        "cpu_temp": sensor.get("cpu_temp_c"),
        "gpu_temp": sensor.get("gpu_temp_c"),
        "batt_pct": sensor.get("batt_pct"),
        "mem_pressure": sensor.get("mem_pressure_gb"),
        "compute_unit": "gpu",
        "batch_size": batch_size,
        "tokens_generated": last_resp.generation_tokens if last_resp else 0,
        "tok_sec": tok_sec,
        "reward": meter.get_reward(),
    })

    last_state = state
    last_action = action_idx
    last_tok_sec = tok_sec

    cpu = sensor.get("cpu_temp_c", 0) or 0
    reward_str = f"{meter.get_reward():.1f}" if meter.get_reward() else "pending"
    results.append({
        "round": i + 1,
        "batch_size": batch_size,
        "tok_sec": round(tok_sec, 1),
        "cpu_temp_c": round(cpu, 1),
        "policy_calls": policy.call_count,
        "wall_sec": round(wall, 2),
    })

    print(
        f"{i+1:<5} {batch_size:<7} {tok_sec:<10.1f} {cpu:<9.1f} {reward_str:<12} {wall:.2f}s",
        flush=True,
    )

buf.stop()

print("\n" + "=" * 60, flush=True)
print("RESULTS SUMMARY", flush=True)
print("=" * 60, flush=True)
print(f"Total policy updates: {policy.call_count}", flush=True)
print(f"Final UCB scores:", flush=True)
final_state = StateBuilder(buf).get()
scores = policy.ucb_scores(final_state)
for a in range(N_ACTIONS):
    bs = batch_size_for(a)
    print(f"  batch_size={bs}: {scores[a]:.3f}", flush=True)

import json
out_path = os.path.join(os.path.dirname(__file__), "results", "phase4_benchmark.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "rounds": results,
        "policy_calls": policy.call_count,
        "ucb_scores": {str(batch_size_for(a)): round(float(scores[a]), 4) for a in range(N_ACTIONS)},
    }, f, indent=2)
print(f"\nResults saved to {out_path}", flush=True)
print("DONE", flush=True)
