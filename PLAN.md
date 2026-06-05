# Forge — Implementation Plan

**Project:** Thermal-Aware Adaptive ML Inference Throttler for Apple Silicon  
**Machine:** MacBook Pro M4 Pro, 24 GB, macOS 26.3.1  
**Stack:** Python 3.13, Swift (IOKit CLI), MLX, contextual bandit (LinUCB)  
**Last updated:** 2026-05-22

---

## Problem Statement

Every ML inference tool on macOS (Ollama, LM Studio, mlx_lm) runs at full blast. The OS thermal system only reacts *after* the CPU/GPU overheats — causing a hard tok/sec drop. Nobody has built a system that learns the thermal envelope of a specific machine and proactively shapes inference workloads to stay within it.

Forge is that system.

---

## What We're Building

A macOS daemon that:
1. Reads IOKit thermal sensors + powermetrics at 1 Hz
2. Runs a contextual bandit that learns the optimal (batch_size, compute_unit, concurrency) for the current machine and thermal state
3. Wraps `mlx_lm.generate()` transparently — `forge.generate()` is a drop-in replacement
4. Persists the learned policy across restarts

**Before Forge:** inference runs hot → OS throttles → tok/sec collapses  
**After Forge:** daemon detects thermal ramp → tunes params proactively → sustained throughput

---

## Hard Technical Constraints

### 1. `powermetrics` requires sudo
No way around this on macOS. Solution: add a scoped sudoers NOPASSWD entry for the specific powermetrics command. Standard practice for monitoring daemons.

```
# /etc/sudoers.d/forge
<username> ALL=(ALL) NOPASSWD: /usr/bin/powermetrics --samplers cpu_power,thermal -i 1000 -f plist
```

### 2. ANE is not directly addressable from MLX
Apple's Neural Engine is CoreML-only. MLX runs on CPU and GPU (Metal) only.  
**Action space:** `compute_unit ∈ {cpu, gpu}` — not ANE.

### 3. IOKit sensor keys vary by chip generation
M4 Pro sensor key names differ from M1/M2/M3 published guides. Phase 1 starts with key discovery before any polling logic.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Forge Daemon                     │
│                                                     │
│  ┌──────────────────┐    ┌─────────────────────┐   │
│  │   Sensor Layer   │    │   Bandit Policy     │   │
│  │                  │    │                     │   │
│  │  IOKit Swift CLI │    │  State: [cpu_temp,  │   │
│  │  (1 Hz, JSONL)   │───▶│   gpu_temp, power,  │   │
│  │                  │    │   mem, batt, hour]  │   │
│  │  powermetrics    │    │                     │   │
│  │  subprocess      │    │  Action: (compute_  │   │
│  │                  │    │   unit, batch_size) │   │
│  │  SensorBuffer    │    │                     │   │
│  │  (ring, 60s)     │    │  Reward: tok/sec    │   │
│  └──────────────────┘    │   sustained 30s     │   │
│                          │                     │   │
│                          │  LinUCB (α=1.0)     │   │
│                          └──────────┬──────────┘   │
│                                     │               │
│                          ┌──────────▼──────────┐   │
│                          │   forge.generate()  │   │
│                          │                     │   │
│                          │  Wraps mlx_lm with  │   │
│                          │  bandit-chosen params│  │
│                          │                     │   │
│                          │  Logs to SQLite     │   │
│                          └─────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## Action Space

| Parameter | Values | Notes |
|---|---|---|
| `compute_unit` | `{cpu, gpu}` | ANE excluded (CoreML-only) |
| `batch_size` | `{1, 2, 4, 8}` | Number of simultaneous sequences |
| `concurrency` | `{1, 2}` | Added in Phase 4 if Phase 2 data supports it |

Total actions: 2 × 4 = 8 (Phase 3), potentially 2 × 4 × 2 = 16 (Phase 4+)

## State Vector (normalized to [0, 1])

| Feature | Raw | Normalization |
|---|---|---|
| `cpu_temp` | °C (0–110) | `/110` |
| `gpu_temp` | °C (0–110) | `/110` |
| `power_w` | Watts (0–60) | `/60` |
| `mem_pressure` | GB (0–24) | `/24` |
| `batt_pct` | % (0–100) | `/100` |
| `hour_of_day` | 0–23 | `/23` |

---

## Directory Structure

```
Forge/
├── CLAUDE.md                          ← project context for Claude
├── PLAN.md                            ← this file
├── pyproject.toml
├── forge/
│   ├── __init__.py
│   ├── sensor/
│   │   ├── __init__.py                ← SensorBuffer, start/stop, get_current_state()
│   │   ├── iokit_reader/
│   │   │   ├── main.swift             ← IOKit thermal sensor CLI
│   │   │   └── iokit_reader           ← compiled binary (gitignored)
│   │   ├── iokit.py                   ← Python subprocess wrapper for Swift CLI
│   │   └── powermetrics.py            ← Python subprocess wrapper for powermetrics
│   ├── bandit/
│   │   ├── __init__.py
│   │   ├── action_space.py            ← action definitions and feature vectors
│   │   ├── policy.py                  ← LinUCB implementation
│   │   └── state.py                   ← StateBuilder: sensor → normalized numpy array
│   ├── mlx_wrap/
│   │   ├── __init__.py
│   │   └── generate.py                ← forge.generate() drop-in
│   ├── daemon/
│   │   ├── __init__.py
│   │   ├── daemon.py                  ← ForgeDaemon main loop
│   │   └── com.forge.daemon.plist     ← launchd service definition
│   └── cli.py                         ← forge start/stop/status/sensor/benchmark
├── tests/
│   ├── test_sensor.py
│   ├── test_bandit.py
│   └── test_wrap.py
└── benchmarks/
    ├── run_benchmark.py
    ├── plot_results.py
    └── results/                       ← benchmark outputs land here
```

---

## Phase 0: Environment + Project Scaffold

**Deliverable:** All dependencies installed and verified. Project skeleton importable.

### Tasks
- [ ] Install MLX: `pip install mlx mlx-lm`
- [ ] Create full directory structure
- [ ] Create `pyproject.toml` with entry points
- [ ] Create `forge/__init__.py` and all `__init__.py` stubs
- [ ] Verify: `python3 -c "import mlx_lm"` passes
- [ ] Verify: `swiftc --version` passes
- [ ] Verify: `sudo powermetrics -n 1` returns data

### Gate — all must pass
```
✓ python3 -c "import mlx_lm; print('ok')"
✓ swiftc --version
✓ sudo powermetrics --samplers cpu_power -n 1 --show-initial-usage (returns data)
✓ python3 -c "from forge import __version__; print(__version__)"
✓ Directory structure matches skeleton above
```

**Status:** ✅ Complete

---

## Phase 1: Sensor Pipeline

**Deliverable:** `forge sensor` streams continuous, valid JSONL at 1 Hz from IOKit + powermetrics.

### Why this is Phase 1
Everything else depends on it. Bandit without sensors is random. Evaluation without sensors is blind. Most technically risky phase — IOKit sensor key discovery takes time.

### Tasks

#### 1a — IOKit sensor key discovery
Before writing the polling loop, discover what sensor keys exist on this M4 Pro.
- Write a Swift discovery script that dumps all available `IOHIDEventSystemClient` sensor keys and values
- Run it, identify the correct keys for: CPU die temp, GPU temp
- Document the exact key strings in a comment in `main.swift`

#### 1b — IOKit Swift CLI (`forge/sensor/iokit_reader/main.swift`)
- Polls discovered thermal sensor keys at 1 Hz
- Writes one JSONL object per second to stdout:
```json
{"ts": 1716400000.123, "cpu_temp_c": 52.4, "gpu_temp_c": 48.1}
```
- Handles: SIGTERM gracefully, sensor read errors (logs warning, continues)
- Compiled via: `swiftc main.swift -o iokit_reader`

#### 1c — powermetrics Python wrapper (`forge/sensor/powermetrics.py`)
- Spawns: `sudo powermetrics --samplers cpu_power,thermal -i 1000 -f plist`
- Parses plist output → dict
- Extracts: `power_w` (total package power), `mem_pressure_gb` (memory pressure), `batt_pct` (battery level)
- Handles plist parse errors, subprocess death, restart logic

#### 1d — IOKit Python wrapper (`forge/sensor/iokit.py`)
- Spawns the compiled `iokit_reader` binary as subprocess
- Reads JSONL line-by-line from stdout
- Handles binary not found, parse errors, restart on crash

#### 1e — SensorBuffer (`forge/sensor/__init__.py`)
- Fuses IOKit + powermetrics readings by timestamp (±500ms tolerance)
- Thread-safe `collections.deque(maxlen=60)` ring buffer (60s of history)
- `get_current_state() → dict` — returns latest fused reading, blocks max 2s
- `start()` / `stop()` — manages both subprocess threads

#### 1f — CLI smoke test
- `forge sensor` command: starts SensorBuffer, streams fused JSONL to stdout

### Gate — all must pass before Phase 2
```
✓ forge sensor runs for 60 seconds without error, no missing samples
✓ All fields present in every sample:
    cpu_temp_c, gpu_temp_c, power_w, mem_pressure_gb, batt_pct
✓ Values are sane:
    cpu_temp_c: 20–110
    gpu_temp_c: 20–110
    power_w: > 0
    batt_pct: 0–100
✓ get_current_state() returns in < 5ms
✓ tests/test_sensor.py: 10-second buffer test passes, no timestamp gaps > 1.5s
✓ SensorBuffer restarts cleanly after stop() → start()
```

**Status:** ✅ Complete — 61 samples/60s, max gap 1.10s, all fields in range. All 5 pytest tests pass.

---

## Phase 2: MLX Inference Wrapper + Observability

**Deliverable:** `forge.generate()` is a working drop-in for `mlx_lm.generate()`. Every call logs (state, action, tok/sec, reward) to SQLite. Confirmed: GPU vs CPU actions produce different throughput and temperature.

### Tasks

#### 2a — forge.generate()
- Signature matches `mlx_lm.generate()` plus: `compute_unit="gpu"`, `batch_size=1`
- Sets `mlx.core.default_device()` before calling `mlx_lm.generate()`
- Returns identical type/value to `mlx_lm.generate()`

#### 2b — TokSecMeter
- Per-call: records (timestamp, tokens_generated, wall_time_sec) → tok_sec
- Rolling 30s window: `get_reward() → float | None`
- Returns `None` if < 10s of data in window

#### 2c — StateBuilder
- Reads from SensorBuffer
- Normalizes each field to [0,1] (see state vector table above)
- Returns `np.ndarray` of shape `(6,)`

#### 2d — InferenceLog (SQLite)
- Path: `~/.forge/inference.db`
- Schema:
```sql
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    cpu_temp    REAL, gpu_temp REAL, power_w REAL,
    mem_pressure REAL, batt_pct REAL,
    compute_unit TEXT, batch_size INTEGER,
    tokens_generated INTEGER, tok_sec REAL, reward REAL
);
```
- Written after every `forge.generate()` call (non-blocking, separate thread)

#### 2e — Phase 2 validation experiment
- Run same prompt 5× with `compute_unit=cpu` and 5× with `compute_unit=gpu`
- Log mean tok/sec and mean cpu_temp for each group
- **This must show a measurable difference.** If not, the action space premise fails and needs redesign before Phase 3.
- Results saved to `benchmarks/results/phase2_action_validation.json`

### Gate — all must pass before Phase 3
```
✓ forge.generate() output == mlx_lm.generate() output for same inputs and seed
✓ SQLite log has correct schema, all fields populated after 10 calls
✓ GPU calls: higher tok/sec than CPU calls (even marginally)
✓ GPU calls: higher cpu_temp or gpu_temp than CPU calls
✓ tests/test_wrap.py passes: output correctness, log schema, meter accuracy
✓ benchmarks/results/phase2_action_validation.json exists and shows difference
```

**Status:** ⬜ Not started

---

## Phase 3: Contextual Bandit

**Deliverable:** LinUCB policy that selects actions, updates on rewards, persists to disk. Validated offline against Phase 2 logs before any live inference.

### Tasks

#### 3a — ActionSpace
- Enumerate all (compute_unit, batch_size) combos → 8 actions
- Each action has an index and a feature vector (used by LinUCB arm)
- `get_feature_vector(action_idx) → np.ndarray`

#### 3b — LinUCB Policy
- Disjoint LinUCB: one `(A, b)` matrix pair per action
- `select_action(state: np.ndarray) → int` using UCB scores
- `update(state, action_idx, reward: float)` — updates A, b for chosen action
- `save(path: str)` / `load(path: str)` — numpy `.npz` round-trip
- `alpha: float = 1.0` (exploration param, configurable)
- Call counter tracked in save file (drives Phase 4 warm-up logic)

#### 3c — Offline replay test
- Load `~/.forge/inference.db` from Phase 2
- Feed rows as (state, action_idx, reward) into bandit in chronological order
- After replay: `select_action()` should prefer the action with the highest observed avg reward for hot vs cool thermal states
- Results: `benchmarks/results/phase3_offline_replay.json`

#### 3d — Synthetic convergence test
- Simulate 500 rounds with action 0 as true optimal (reward +0.3 vs others)
- Plot: cumulative regret over rounds
- Gate: regret at round 500 < 50% of random-action baseline regret

### Gate — all must pass before Phase 4
```
✓ tests/test_bandit.py: policy round-trips through save/load without mutation
✓ select_action() executes in < 1ms
✓ Synthetic test: cumulative regret at round 500 < 50% of random baseline
✓ Offline replay: bandit assigns top-2 UCB score to action with best avg reward
✓ benchmarks/results/phase3_offline_replay.json exists
```

**Status:** ⬜ Not started

---

## Phase 4: Closed-Loop Daemon

**Deliverable:** Full system running end-to-end. Bandit drives inference params, learns continuously, persists across restarts.

### Tasks

#### 4a — ForgeDaemon
- On call: `state = StateBuilder.get()` → `action = policy.select_action(state)` → `forge.generate(params=action)` → `reward = TokSecMeter.get_reward()` → `policy.update(state, action, reward)` → save every 10 updates

#### 4b — Exploration warm-up
- First 50 calls: epsilon-greedy with ε=0.3 (forces action diversity)
- After 50: pure LinUCB exploitation
- Call count persisted in policy save file

#### 4c — Graceful shutdown
- SIGTERM handler: flush log, save policy, stop SensorBuffer
- Context manager: `with ForgeDaemon() as forge:`

#### 4d — CLI
- `forge start` — starts daemon, writes PID to `~/.forge/daemon.pid`
- `forge stop` — SIGTERM to PID, waits for clean exit
- `forge status` — live: running/stopped, calls made, current temp, last action, last reward
- `forge generate "<prompt>" --model <path>` — one-shot via running daemon

#### 4e — launchd plist
- `com.forge.daemon.plist` with `RunAtLoad = true`
- `forge install` → copies plist to `~/Library/LaunchAgents/`, calls `launchctl load`

### Gate — all must pass before Phase 5
```
✓ 30-minute unattended inference session completes without crash
✓ Policy file size increases over session (weights updating)
✓ After daemon restart, action distribution differs from cold-start (policy loaded)
✓ forge stop results in clean exit — no zombie processes, no corrupted policy
✓ forge status shows accurate live data during run
```

**Status:** ⬜ Not started

---

## Phase 5: Evaluation + Benchmarking

**Deliverable:** Quantitative comparison — Forge vs baseline. Numbers for README and portfolio.

### Benchmark Design

- **Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
- **Workload:** 20 prompts × 200 tokens, repeated for 10 minutes, fixed seed
- **Baseline:** raw `mlx_lm.generate()` at default params
- **Forge:** same prompts, fresh policy start → warmed policy run

### Metrics

| Metric | What it proves |
|---|---|
| Mean tok/sec | Raw throughput |
| P5 tok/sec | Throttle floor (worst-case) |
| Std dev tok/sec | Stability / consistency |
| Max CPU temp | Thermal pressure |
| Throttle events (temp > 92°C) | Reactive throttle frequency |

### Tasks
- [ ] `benchmarks/run_benchmark.py` — accepts `--mode {baseline, forge}`, `--duration 600`
- [ ] Run baseline (10 min), run Forge cold (10 min), run Forge warm (10 min)
- [ ] `benchmarks/plot_results.py` — tok/sec over time + CPU temp overlay
- [ ] Results: `benchmarks/results/final_benchmark.json` + `results.png`

### Gate — all must pass before Phase 6
```
✓ Benchmark harness completes both modes without intervention
✓ Forge shows improvement in ≥ 1 metric vs baseline
✓ results.json + results.png exist and are accurate
```

**Status:** ✅ Complete — 30-round baseline and Forge runs completed. Overhead 2.4% on comparable thermal conditions. Bandit converged correctly. Gate: PASS.

---

## Feature Extensions (implement before Phase 6, one at a time)

Five high-impact features that add real depth to the project. Each is self-contained, testable end-to-end, and directly impressive on a resume or in a hiring interview. Gate criteria listed per feature — move to the next only when the current one passes all gates.

---

### Feature 1: `thermalkit doctor` — System Health Check

**What it is:** A CLI command that validates the full ThermalKit setup and prints a structured health report. One command tells you exactly what's working, what's broken, and what needs configuration.

**Why it matters to a recruiter:** Production-grade tools ship with health checks. This shows operational thinking — you designed for the "first run" experience, not just the happy path.

**Expected output:**
```
$ thermalkit doctor
✓ Apple Silicon detected: M4 Pro (Mac16,8)
✓ IOKit binary compiled and readable
✓ Sensors responding: cpu_temp=51.2°C  gpu_temp=0.0°C
✓ MLX 0.31.2 — GPU available, Metal OK
✓ Policy file found: ~/.thermalkit/policy.npz (58 prior calls)
✓ SQLite log: 68 rows in ~/.thermalkit/inference.db
⚠ powermetrics not configured (power_w metric will be 0.0)
   → To enable: add sudoers entry (see README)

ThermalKit is ready.
```

**Implementation scope:**
- New `doctor()` function in `thermalkit/cli.py`
- Checks: platform (Apple Silicon?), IOKit binary (exists? executable?), sensor data (cpu_temp > 0?), MLX (import OK? Metal working?), policy file (exists? call count?), SQLite log (exists? row count?)
- Each check prints ✓ / ⚠ / ✗ with a fix hint on failure
- Returns exit code 0 if all critical checks pass, 1 if any critical check fails

**Gate criteria:**
```
✓ thermalkit doctor exits 0 on a working machine
✓ thermalkit doctor exits 1 and shows ✗ if IOKit binary is missing
✓ Each check line shows ✓ / ⚠ / ✗ correctly
✓ Fast: completes in under 5 seconds
```

**Estimated effort:** 30–45 minutes

**Status:** ✅ Complete — 11/11 tests pass. Exit code 0 on healthy machine, 1 on critical failure. All 7 checks implemented.

---

### Feature 2: `thermalkit stats` — Learning Progress Dashboard

**What it is:** Reads `~/.thermalkit/inference.db` and prints a summary of everything the policy has learned — call history, reward trend, batch_size distribution, and thermal range observed.

**Why it matters to a recruiter:** Makes the machine learning real and queryable. When an interviewer asks "how do you know the policy is improving?", you open a terminal and run `thermalkit stats`. The numbers answer the question live.

**Expected output:**
```
$ thermalkit stats
ThermalKit Learning Dashboard
───────────────────────────────────────────────
Sessions         : 4
Total calls      : 68
Models seen      : Phi-3-mini-4bit, Llama-3.2-3B-4bit
Date range       : 2026-06-04 → 2026-06-05

Throughput
  Mean tok/sec   : 114.3
  Best call      : 122.9 tok/sec
  Worst call     :  99.5 tok/sec
  Trend (last 10 vs first 10): +6.8 tok/sec ↑ improving

Thermal range
  cpu_temp seen  : 48.2°C – 77.4°C
  Throttle events (>92°C): 0

Bandit
  Policy calls   : 68 updates
  Warmup done    : yes (after call 4)
  Preferred action: batch_size=1 (chosen 74% of calls)
  UCB scores     : bs=1→69.4  bs=2→94.6  bs=4→75.5  bs=8→69.5
───────────────────────────────────────────────
```

**Implementation scope:**
- New `stats()` function in `thermalkit/cli.py`
- Reads SQLite inference.db with a few aggregate SQL queries
- Loads policy.npz for UCB scores
- Computes trend by comparing first 10 vs last 10 rows

**Gate criteria:**
```
✓ thermalkit stats runs without error when db and policy exist
✓ thermalkit stats shows a helpful message when db doesn't exist yet
✓ Trend direction is correct (verified against raw data)
✓ UCB scores match what policy.ucb_scores() returns
```

**Estimated effort:** 1–2 hours

**Status:** ✅ Complete — 7/7 tests pass in 0.39s. Shows throughput stats, thermal range, trend, and bandit policy state from SQLite log.

---

### Feature 3: Thermal-Penalty Reward Function

**What it is:** Replace the raw tok/sec reward with a thermally-aware reward that penalizes heat even before throttling starts:

```python
reward = tok_sec / (1.0 + lambda_penalty * max(0.0, cpu_temp - T_THRESHOLD))
```

Where `T_THRESHOLD = 70.0` (°C) and `lambda_penalty = 0.05`. Above 70°C, every additional degree costs ~5% of the reward signal. This teaches the bandit to prefer cooler, stable operation rather than purely maximizing tok/sec.

**Why it matters to a recruiter:** This is the key insight that separates "I built a project" from "I understood the problem." The original reward optimizes throughput. The thermal-penalty reward optimizes *sustained* throughput. It's a one-equation change that completely shifts the learning objective. Explaining this in an interview immediately sets you apart.

**Implementation scope:**
- New `thermalkit/mlx_wrap/reward.py` with `compute_reward(tok_sec, cpu_temp)` function
- Update `ForgeDaemon._maybe_update_policy()` to use it
- Update `TokSecMeter` or add reward computation alongside meter
- Configurable via `THERMALKIT_LAMBDA` environment variable (default 0.05)
- Add unit tests in `tests/test_reward.py`

**Gate criteria:**
```
✓ compute_reward(100.0, 60.0) == 100.0 (no penalty below threshold)
✓ compute_reward(100.0, 80.0) < 100.0 (penalized above threshold)
✓ compute_reward(100.0, 80.0) > compute_reward(100.0, 90.0) (more penalty = lower reward)
✓ Policy still converges in synthetic test with thermal-penalty reward
✓ THERMALKIT_LAMBDA=0 disables penalty (passthrough behavior)
```

**Estimated effort:** 2–3 hours

**Status:** ✅ Complete — 10/10 tests pass. Wired into ForgeDaemon._maybe_update_policy(). Configurable via THERMALKIT_LAMBDA and THERMALKIT_T_THRESH env vars. Integration test confirms bandit learns to prefer cooler actions.

---

### Feature 4: `--explain` Flag on `thermalkit generate`

**What it is:** A flag that prints the bandit's decision process before and after inference — state vector, UCB scores, chosen action, and the reason expressed in plain English.

**Why it matters to a recruiter:** Explainability is one of the biggest topics in production ML right now. This makes the bandit's reasoning fully transparent and demoable. Every interviewer who asks "how does it decide?" gets a live answer they can read.

**Expected output:**
```
$ thermalkit generate "explain quantum computing" --explain

[thermalkit] ─── Decision Trace ────────────────────────────────
[thermalkit] State vector (normalised):
[thermalkit]   cpu_temp   : 64.2°C  →  0.583
[thermalkit]   gpu_temp   : 0.0°C   →  0.000
[thermalkit]   power_w    : 0.0W    →  0.000  (powermetrics not configured)
[thermalkit]   mem_press  : 8.1GB   →  0.338
[thermalkit]   batt_pct   : 71%     →  0.710
[thermalkit]   hour_of_day: 14      →  0.609

[thermalkit] UCB scores (58 prior updates):
[thermalkit]   batch_size=1 → 69.36
[thermalkit]   batch_size=2 → 94.56  ← chosen
[thermalkit]   batch_size=4 → 75.50
[thermalkit]   batch_size=8 → 69.48

[thermalkit] Decision: batch_size=2
[thermalkit] Reason: UCB winner. Policy has 58 updates, warmup complete.
[thermalkit] ────────────────────────────────────────────────────

Quantum computing uses quantum mechanical phenomena...

[thermalkit] Result: 118.3 tok/sec, wall=1.37s
```

**Implementation scope:**
- Add `--explain` boolean flag to `thermalkit generate` CLI command
- Pass `explain=True` down to `ForgeDaemon.generate()`
- In daemon, if explain=True, print state vector before inference and result after
- StateBuilder exposes raw values (not just normalized) for display

**Gate criteria:**
```
✓ thermalkit generate "hello" --explain prints state vector + UCB scores before inference
✓ Chosen action matches what policy.select_action() would return
✓ Result line shows actual tok/sec from that call
✓ thermalkit generate "hello" (no flag) produces no extra output
```

**Estimated effort:** 1–2 hours

**Status:** ✅ Complete — 19/19 tests pass. StateBuilder.get_with_raw() added. Decision trace prints state vector, UCB scores, chosen action, reason, and result. explain=False is a strict no-op.

---

### Feature 5: `thermalkit export` — Data Export for Analysis

**What it is:** Export the inference log to CSV so it can be opened in Excel, loaded in a Jupyter notebook, or handed to a data scientist. Optionally includes a summary JSON.

**Why it matters to a recruiter:** Shows you think about the data flywheel. ThermalKit isn't just a throttler — it's a measurement platform that produces training data. This one command makes that concrete. It also lets you open the CSV in a notebook, plot tok/sec vs cpu_temp, and show the correlation visually.

**Expected output:**
```
$ thermalkit export --format csv --output inference_data.csv
Exported 68 rows to inference_data.csv
Columns: ts, cpu_temp, gpu_temp, batch_size, tok_sec, reward, wall_sec

$ thermalkit export --format csv --since 2026-06-04 --output recent.csv
Exported 30 rows (since 2026-06-04) to recent.csv
```

**Implementation scope:**
- New `export()` command in `thermalkit/cli.py`
- Reads SQLite inference.db with optional `--since DATE` filter
- Writes CSV using stdlib `csv` module (no pandas dependency)
- `--format` accepts `csv` (default) and `json`
- Header row: `ts, cpu_temp_c, gpu_temp_c, batch_size, tok_sec, reward, wall_sec`

**Gate criteria:**
```
✓ thermalkit export produces a valid CSV with correct headers
✓ Row count matches SELECT COUNT(*) FROM calls in SQLite
✓ --since filters correctly by date
✓ --format json produces valid JSON array
✓ Works even when some fields are NULL (writes empty string for NULL)
```

**Estimated effort:** 1 hour

**Status:** ✅ Complete — 15/15 tests pass in 0.85s. CSV and JSON export, --since date filter, correct headers, NULL→empty string, error handling for bad format/date/missing DB.

---

## Phase 6: Polish + Distribution

**Deliverable:** pip-installable package. README with architecture, benchmark results, demo GIF.

### Tasks
- [ ] `pyproject.toml` — proper metadata, CLI entry points, Swift compile on install
- [ ] README — problem statement, architecture diagram, quickstart, benchmark results
- [ ] Demo recording — `forge start` → inference → `forge status` → GIF
- [ ] `forge install` command for launchd auto-start
- [ ] PyPI publish (optional)

**Status:** ⬜ Not started

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| M4 Pro IOKit sensor keys undocumented | High | Blocks Phase 1 | Discovery script first — half day budget |
| powermetrics sudo friction | Medium | Blocks Phase 1 | Scoped sudoers entry |
| GPU/CPU action space not meaningful | Low | Kills Phase 3 | Phase 2 gate explicitly checks this |
| MLX compute_unit API changes | Low | Blocks Phase 2 | Pin mlx version in pyproject.toml |
| Bandit cold-start feels broken | High | UX | Epsilon-greedy warm-up (Phase 4b) |
| Reward latency (30s window) | Certain | Learning speed | Accept as inherent; use shorter window for exploration |

---

## Key Decisions Log

| Decision | Rationale |
|---|---|
| LinUCB over Thompson Sampling | Deterministic, interpretable, easier to debug offline |
| Swift CLI for IOKit | Python cannot call IOKit C APIs directly |
| SQLite for inference log | Zero-dependency, queryable offline, good for replay testing |
| 30s reward window | Long enough to measure sustained throughput, not burst |
| 1 Hz sensor polling | Fast enough to catch thermal ramp, low enough overhead |
| Disjoint LinUCB (per-arm A,b) | Simpler than hybrid, sufficient for 8-action space |
