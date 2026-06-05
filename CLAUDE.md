# Forge — Thermal-Aware Adaptive ML Inference Throttler for Apple Silicon

## What This Is

Forge is an adaptive inference middleware for macOS Apple Silicon. It wraps MLX inference calls and dynamically tunes execution parameters (batch size, compute unit, concurrency) based on real-time thermal and power data — using online learning (contextual bandit) to proactively prevent thermal throttling before it happens.

## The Core Problem

Every ML inference tool on macOS (Ollama, LM Studio, MLX community scripts) runs at full blast. The machine has no awareness that it's about to thermally throttle. The system only reacts after throttling begins — causing sudden quality degradation (lower tok/sec). Forge learns machine-specific thermal behavior and shapes workloads proactively.

## Architecture

```
Forge Daemon
├── IOKit Reader (Swift CLI, 1 Hz loop)        — CPU/GPU temp, die temp
├── powermetrics subprocess                     — power draw, efficiency/perf core load
├── Bandit Policy
│   ├── State:  (cpu_temp, mem_pressure, batt%, time_of_day)
│   ├── Action: (batch_size, compute_unit [CPU/GPU/ANE], concurrency)
│   └── Reward: tok/sec sustained over 30s window
└── MLX Inference Wrap (forge.generate())       — drop-in replacement for mlx_lm.generate()
```

## Tech Stack

- **Language**: Python (core daemon, bandit, MLX wrap) + Swift (IOKit sensor CLI)
- **ML Framework**: MLX (Apple Silicon native)
- **Sensor Access**: IOKit via Swift CLI subprocess, `powermetrics` subprocess
- **Learning**: Contextual bandit (likely LinUCB or Thompson Sampling)
- **Interface**: Drop-in `forge.generate()` wrapping `mlx_lm.generate()`
- **Platform**: macOS only, Apple Silicon required

## Key Design Decisions

- **1 Hz sensor polling**: Fast enough to catch thermal ramp, slow enough to not skew measurements
- **30s reward window**: Long enough to measure sustained throughput, not burst
- **Swift for IOKit**: Python cannot access IOKit directly; Swift CLI subprocess bridges this
- **No code changes required**: Any MLX model benefits by swapping `mlx_lm.generate` → `forge.generate`
- **Contextual bandit, not RL**: Stateless per-episode learning — simpler, faster convergence than full RL

## Novelty / Why This Hasn't Been Built

Requires three domains simultaneously:
1. Apple Silicon thermal internals (IOKit)
2. Online learning (contextual bandit)
3. MLX inference orchestration

No open-source project spans all three. Apple's Battery & Thermal Engineering and Core ML Platform teams are the only groups likely doing this internally.

## Phases (TBD — fill in as development progresses)

- [ ] Phase 1: Sensor pipeline — IOKit Swift CLI + powermetrics → structured JSON at 1 Hz
- [ ] Phase 2: Baseline MLX wrap — forge.generate() passthrough with timing + reward logging
- [ ] Phase 3: Bandit policy — LinUCB or Thompson Sampling, action space defined
- [ ] Phase 4: Closed loop — bandit drives inference parameters, reward feeds back
- [ ] Phase 5: Evaluation — compare tok/sec sustained vs unthrottled baseline

## File Structure (Expected)

```
forge/
├── daemon/
│   ├── sensor_reader.py       — orchestrates IOKit + powermetrics polling
│   ├── iokit_reader/          — Swift CLI source
│   └── powermetrics.py        — subprocess wrapper
├── bandit/
│   ├── policy.py              — contextual bandit implementation
│   └── state.py               — state vector construction
├── mlx_wrap/
│   └── generate.py            — forge.generate() drop-in
├── config.py
└── main.py                    — daemon entry point
```

## Environment

- macOS, Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- MLX installed
- Xcode CLI tools (for Swift compilation)
- `powermetrics` requires sudo or entitlement
