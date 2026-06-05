import sys
import typer

app = typer.Typer(
    name="thermalkit",
    help="Thermal-aware adaptive ML inference throttler for Apple Silicon.",
    no_args_is_help=True,
)

# ── ANSI colour helpers (work in any macOS terminal) ─────────────────────────
def _ok(msg: str)   -> str: return f"\033[32m✓\033[0m  {msg}"
def _warn(msg: str) -> str: return f"\033[33m⚠\033[0m  {msg}"
def _fail(msg: str) -> str: return f"\033[31m✗\033[0m  {msg}"


@app.command()
def doctor():
    """Validate the ThermalKit setup and print a structured health report."""
    import os
    import platform
    import sqlite3
    import subprocess
    import time
    from pathlib import Path

    passed   = 0
    warnings = 0
    failed   = 0
    lines    = []

    def ok(msg, detail=""):
        nonlocal passed
        passed += 1
        lines.append(_ok(msg) + (f"\n     {detail}" if detail else ""))

    def warn(msg, fix=""):
        nonlocal warnings
        warnings += 1
        lines.append(_warn(msg) + (f"\n     → {fix}" if fix else ""))

    def fail(msg, fix=""):
        nonlocal failed
        failed += 1
        lines.append(_fail(msg) + (f"\n     → {fix}" if fix else ""))

    print("\nThermalKit — System Health Check", flush=True)
    print("─" * 45, flush=True)

    # ── 1. Apple Silicon ──────────────────────────────────────────────────────
    machine = platform.machine()
    chip = _detect_chip()
    if machine == "arm64":
        ok(f"Apple Silicon detected: {chip}")
    else:
        fail(
            f"Not Apple Silicon ({machine} / {chip})",
            "ThermalKit requires an Apple Silicon Mac (M1/M2/M3/M4).",
        )

    # ── 2. IOKit binary ───────────────────────────────────────────────────────
    from thermalkit.sensor.iokit import _BINARY
    if not _BINARY.exists():
        fail(
            "IOKit binary not found",
            f"Run: cd {_BINARY.parent} && swiftc main.swift -o iokit_reader",
        )
    elif not os.access(str(_BINARY), os.X_OK):
        fail(
            f"IOKit binary not executable: {_BINARY}",
            f"Run: chmod +x {_BINARY}",
        )
    else:
        ok(f"IOKit binary compiled and executable", str(_BINARY))

    # ── 3. Sensor readings ────────────────────────────────────────────────────
    try:
        from thermalkit.sensor import SensorBuffer
        buf = SensorBuffer()
        buf.start()
        time.sleep(2.5)
        state = buf.get_current_state(timeout=1.0)
        buf.stop()

        if state and state.get("cpu_temp_c") and state["cpu_temp_c"] > 0:
            cpu = state["cpu_temp_c"]
            gpu = state.get("gpu_temp_c", 0.0)
            batt = state.get("batt_pct", "?")
            ok(
                f"Sensors responding",
                f"cpu_temp={cpu:.1f}°C  gpu_temp={gpu:.1f}°C  batt={batt}%",
            )
        else:
            fail(
                "Sensors not returning data",
                "Check IOKit binary permissions. The binary must run as subprocess.",
            )
    except Exception as e:
        fail(f"Sensor startup failed: {e}")

    # ── 4. MLX + Metal ────────────────────────────────────────────────────────
    try:
        import mlx.core as mx
        version = mx.__version__
        # Quick Metal smoke-test
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        result = (a + b).tolist()
        if result == [4.0, 6.0]:
            ok(f"MLX {version} — GPU available, Metal OK")
        else:
            warn(f"MLX {version} — Metal computation returned unexpected result")
    except ImportError:
        fail("MLX not installed", "pip install mlx>=0.31 mlx-lm>=0.31")
    except Exception as e:
        fail(f"MLX/Metal error: {e}")

    # ── 5. Policy file ────────────────────────────────────────────────────────
    policy_path = Path.home() / ".thermalkit" / "policy.npz"
    if policy_path.exists():
        try:
            from thermalkit.bandit.policy import LinUCBPolicy
            policy = LinUCBPolicy.load(policy_path)
            ok(
                f"Policy file found",
                f"{policy_path}  ({policy.call_count} prior calls, "
                f"warmup_done={policy._warmup_done})",
            )
        except Exception as e:
            warn(f"Policy file exists but failed to load: {e}", "Delete it to reset: rm ~/.thermalkit/policy.npz")
    else:
        warn(
            "No policy file yet",
            "Run 'thermalkit benchmark' to start learning.",
        )

    # ── 6. SQLite inference log ───────────────────────────────────────────────
    db_path = Path.home() / ".thermalkit" / "inference.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
            conn.close()
            ok(f"Inference log found", f"{db_path}  ({count} rows)")
        except Exception as e:
            warn(f"Inference log exists but unreadable: {e}")
    else:
        warn(
            "No inference log yet",
            "Log is created automatically on first thermalkit generate/benchmark call.",
        )

    # ── 7. powermetrics (optional) ────────────────────────────────────────────
    import shutil
    if shutil.which("powermetrics"):
        # Check sudoers entry — try a zero-sample dry run (fails fast without sudo)
        result = subprocess.run(
            ["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1"],
            capture_output=True, timeout=3,
        )
        if result.returncode == 0:
            ok("powermetrics configured with sudoers NOPASSWD")
        else:
            warn(
                "powermetrics available but requires sudo password",
                "Add sudoers entry for zero-password access (see README). "
                "power_w metric will read as 0.0 until configured.",
            )
    else:
        warn("powermetrics not found", "Should be at /usr/bin/powermetrics on any macOS install.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "\n".join(lines), flush=True)
    print("\n" + "─" * 45, flush=True)

    total = passed + warnings + failed
    status_parts = [f"\033[32m{passed} passed\033[0m"]
    if warnings:
        status_parts.append(f"\033[33m{warnings} warnings\033[0m")
    if failed:
        status_parts.append(f"\033[31m{failed} failed\033[0m")
    print("  ".join(status_parts), flush=True)

    if failed == 0 and warnings == 0:
        print("\nThermalKit is ready.", flush=True)
    elif failed == 0:
        print("\nThermalKit is ready (with warnings — non-critical).", flush=True)
    else:
        print("\nThermalKit has critical issues. Fix the ✗ items above.", flush=True)

    sys.exit(1 if failed > 0 else 0)


def _detect_chip() -> str:
    """Return a human-readable chip name from sysctl."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3
        ).strip()
        return out if out else "Apple Silicon"
    except Exception:
        return "Apple Silicon"


@app.command()
def sensor():
    """Stream live sensor data as JSONL (Ctrl-C to stop)."""
    import json
    import signal
    import sys
    import time

    from thermalkit.sensor import SensorBuffer

    buf = SensorBuffer()
    buf.start()

    def _stop(sig, frame):
        buf.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_ts = None
    while True:
        state = buf.get_current_state()
        if state and state.get("ts") != last_ts:
            last_ts = state["ts"]
            print(json.dumps(state, separators=(",", ":")), flush=True)
        else:
            time.sleep(0.1)


@app.command()
def status():
    """Show sensor readings and bandit policy state."""
    import json
    import time
    from pathlib import Path

    from thermalkit.sensor import SensorBuffer
    from thermalkit.mlx_wrap.state_builder import StateBuilder
    from thermalkit.bandit.policy import LinUCBPolicy
    from thermalkit.bandit.action_space import N_ACTIONS, batch_size_for

    # Sensor snapshot
    buf = SensorBuffer()
    buf.start()
    typer.echo("Reading sensors...")
    time.sleep(2)
    sensor_state = buf.get_current_state() or {}
    buf.stop()

    typer.echo("\n--- Sensors ---")
    typer.echo(json.dumps({
        "cpu_temp_c": sensor_state.get("cpu_temp_c"),
        "gpu_temp_c": sensor_state.get("gpu_temp_c"),
        "batt_pct": sensor_state.get("batt_pct"),
        "mem_pressure_gb": sensor_state.get("mem_pressure_gb"),
    }, indent=2))

    # Policy state
    policy_path = Path.home() / ".thermalkit" / "policy.npz"
    if policy_path.exists():
        policy = LinUCBPolicy.load(policy_path)
        typer.echo("\n--- Bandit Policy ---")
        typer.echo(f"  calls        : {policy.call_count}")
        typer.echo(f"  warmup done  : {policy._warmup_done}")
        typer.echo(f"  policy path  : {policy_path}")
    else:
        typer.echo("\n--- Bandit Policy ---")
        typer.echo("  No policy file found. Run thermalkit generate to start learning.")


@app.command()
def version():
    """Print Forge version."""
    from thermalkit import __version__
    typer.echo(f"thermalkit {__version__}")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Prompt to send to the model."),
    model: str = typer.Option(
        "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "--model", "-m",
        help="HuggingFace model ID or local path.",
    ),
    max_tokens: int = typer.Option(200, "--max-tokens", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    explain: bool = typer.Option(False, "--explain", "-e",
                                  help="Print bandit decision trace before and after inference."),
):
    """Run one inference call with bandit-chosen batch_size."""
    from thermalkit.daemon.daemon import ForgeDaemon

    print(f"[thermalkit] Loading model: {model} (please wait ~20-30s)...", flush=True)
    with ForgeDaemon.from_model(model) as daemon:
        print(f"[thermalkit] Model loaded. Policy calls: {daemon.call_count}", flush=True)
        text = daemon.generate(prompt, verbose=verbose, explain=explain, max_tokens=max_tokens)
        if not verbose and not explain:
            print(text, flush=True)
        elif not verbose:
            # In explain mode the trace already printed; show the response clearly
            print("\n--- Response ---", flush=True)
            print(text, flush=True)
        st = daemon.status()
        if not explain:
            print(
                f"\n[thermalkit] batch_size={st['best_action_batch_size']} | "
                f"cpu={st['cpu_temp_c']}°C | "
                f"reward={st['last_reward']}",
                flush=True,
            )


@app.command()
def benchmark(
    model: str = typer.Option(
        "mlx-community/Phi-3-mini-4k-instruct-4bit",
        "--model", "-m",
    ),
    rounds: int = typer.Option(20, "--rounds", "-r", help="Number of inference calls."),
    max_tokens: int = typer.Option(40, "--max-tokens", "-n"),
):
    """Run N inference calls with the bandit driving batch_size, print a summary."""
    import json
    import time
    from thermalkit.daemon.daemon import ForgeDaemon

    prompts = [
        "Explain thermal throttling in one sentence.",
        "What is Apple Silicon? One sentence.",
        "Name three benefits of online learning algorithms.",
        "What does tok/sec measure in language models?",
        "Describe unified memory in two sentences.",
    ]

    print(f"[thermalkit] Loading model: {model}", flush=True)
    print("[thermalkit] This takes ~20-30s on first run — please wait...", flush=True)
    with ForgeDaemon.from_model(model) as daemon:
        print(f"[thermalkit] Model loaded. Policy has {daemon.call_count} prior calls.", flush=True)
        print(f"[thermalkit] Running {rounds} rounds (max_tokens={max_tokens})\n", flush=True)
        results = []
        for i in range(rounds):
            prompt = prompts[i % len(prompts)]
            t0 = time.perf_counter()
            daemon.generate(prompt, max_tokens=max_tokens)
            wall = time.perf_counter() - t0
            st = daemon.status()
            reward_str = f"{st['last_reward']:.1f} tok/s" if st["last_reward"] else "pending"
            cpu_str = f"{st['cpu_temp_c']:.1f}" if st["cpu_temp_c"] else "n/a"
            entry = {
                "round": i + 1,
                "chosen_batch_size": st["best_action_batch_size"],
                "cpu_temp_c": st["cpu_temp_c"],
                "reward_tok_sec": round(st["last_reward"], 2) if st["last_reward"] else None,
                "wall_sec": round(wall, 2),
                "policy_calls": st["policy_calls"],
            }
            results.append(entry)
            print(
                f"  [{i+1:02d}/{rounds}] "
                f"batch_size={st['best_action_batch_size']} | "
                f"cpu={cpu_str}°C | "
                f"reward={reward_str} | "
                f"wall={wall:.2f}s",
                flush=True,
            )

        print(f"\n[thermalkit] Done. Total policy calls: {daemon.call_count}", flush=True)
        print(json.dumps({"rounds": results, "ucb_scores": daemon.status()["ucb_scores"]}, indent=2), flush=True)


@app.command()
def stats():
    """Show learning progress and inference history from the local database."""
    import sqlite3
    import statistics
    from collections import Counter
    from datetime import datetime
    from pathlib import Path

    db_path = Path.home() / ".thermalkit" / "inference.db"
    # fall back to old path for existing installations
    if not db_path.exists():
        old = Path.home() / ".forge" / "inference.db"
        if old.exists():
            db_path = old

    if not db_path.exists():
        print("No inference log found.", flush=True)
        print("Run 'thermalkit benchmark' to start collecting data.", flush=True)
        sys.exit(0)

    conn = sqlite3.connect(str(db_path))

    rows = conn.execute(
        "SELECT ts, batch_size, tok_sec, cpu_temp, reward FROM calls "
        "WHERE tok_sec IS NOT NULL ORDER BY ts"
    ).fetchall()
    conn.close()

    if not rows:
        print("Inference log exists but contains no completed calls yet.", flush=True)
        sys.exit(0)

    total = len(rows)
    ts_first = rows[0][0]
    ts_last  = rows[-1][0]
    date_first = datetime.fromtimestamp(ts_first).strftime("%Y-%m-%d %H:%M")
    date_last  = datetime.fromtimestamp(ts_last).strftime("%Y-%m-%d %H:%M")

    tok_secs  = [r[2] for r in rows if r[2] is not None]
    cpu_temps = [r[3] for r in rows if r[3] is not None]
    batch_choices = [r[1] for r in rows if r[1] is not None]

    mean_tps = statistics.mean(tok_secs) if tok_secs else 0
    best_tps = max(tok_secs) if tok_secs else 0
    worst_tps = min(tok_secs) if tok_secs else 0

    # Trend: first 10 vs last 10
    first10 = [r[2] for r in rows[:10] if r[2] is not None]
    last10  = [r[2] for r in rows[-10:] if r[2] is not None]
    trend_delta = None
    trend_arrow = ""
    if first10 and last10:
        trend_delta = statistics.mean(last10) - statistics.mean(first10)
        trend_arrow = "↑ improving" if trend_delta > 0.5 else ("↓ declining" if trend_delta < -0.5 else "→ stable")

    min_cpu = min(cpu_temps) if cpu_temps else None
    max_cpu = max(cpu_temps) if cpu_temps else None
    throttle_events = sum(1 for t in cpu_temps if t > 92.0)

    bs_counter = Counter(batch_choices)
    preferred_bs = bs_counter.most_common(1)[0] if bs_counter else (None, 0)
    preferred_pct = (preferred_bs[1] / total * 100) if total > 0 else 0

    # Policy info
    policy_path = Path.home() / ".thermalkit" / "policy.npz"
    policy_info = None
    ucb_line = ""
    if policy_path.exists():
        try:
            from thermalkit.bandit.policy import LinUCBPolicy
            from thermalkit.bandit.action_space import N_ACTIONS, batch_size_for
            import numpy as np
            policy = LinUCBPolicy.load(policy_path)
            norms = np.array([110, 110, 60, 24, 100, 23], dtype=np.float32)
            if cpu_temps:
                state = np.array([
                    statistics.mean(cpu_temps), 0.0, 0.0,
                    4.0, 80.0, 12.0,
                ], dtype=np.float32) / norms
            else:
                state = np.ones(6, dtype=np.float32) * 0.5
            scores = policy.ucb_scores(state)
            score_parts = "  ".join(
                f"bs={batch_size_for(a)}→{scores[a]:.1f}"
                for a in range(N_ACTIONS)
            )
            policy_info = policy
            ucb_line = score_parts
        except Exception:
            pass

    W = 52
    print(flush=True)
    print("ThermalKit Learning Dashboard", flush=True)
    print("─" * W, flush=True)

    print(f"  Total calls      : {total}", flush=True)
    print(f"  Date range       : {date_first}  →  {date_last}", flush=True)

    print(flush=True)
    print("  Throughput", flush=True)
    print(f"    Mean tok/sec   : {mean_tps:.1f}", flush=True)
    print(f"    Best call      : {best_tps:.1f} tok/sec", flush=True)
    print(f"    Worst call     : {worst_tps:.1f} tok/sec", flush=True)
    if trend_delta is not None:
        sign = "+" if trend_delta >= 0 else ""
        print(f"    Trend (L10/F10): {sign}{trend_delta:.1f} tok/sec  {trend_arrow}", flush=True)

    print(flush=True)
    print("  Thermal range", flush=True)
    if min_cpu and max_cpu:
        print(f"    cpu_temp seen  : {min_cpu:.1f}°C – {max_cpu:.1f}°C", flush=True)
    else:
        print(f"    cpu_temp seen  : no data", flush=True)
    print(f"    Throttle events: {throttle_events}  (>92°C)", flush=True)

    print(flush=True)
    print("  Bandit", flush=True)
    if policy_info:
        print(f"    Policy calls   : {policy_info.call_count} updates", flush=True)
        print(f"    Warmup done    : {'yes' if policy_info._warmup_done else 'no'}", flush=True)
    else:
        print(f"    Policy calls   : {total} (estimated from log)", flush=True)
    if preferred_bs[0]:
        print(f"    Preferred action: batch_size={preferred_bs[0]} "
              f"({preferred_pct:.0f}% of calls)", flush=True)
    if ucb_line:
        print(f"    UCB scores     : {ucb_line}", flush=True)

    print("─" * W, flush=True)


@app.command()
def export(
    output: str = typer.Option("inference_data.csv", "--output", "-o",
                                help="Output file path."),
    fmt: str = typer.Option("csv", "--format", "-f",
                             help="Output format: csv or json."),
    since: str = typer.Option(None, "--since",
                               help="Filter rows on or after this date (YYYY-MM-DD)."),
):
    """Export the inference log to CSV or JSON for analysis."""
    import csv
    import json
    import sqlite3
    from datetime import datetime
    from pathlib import Path

    db_path = Path.home() / ".thermalkit" / "inference.db"
    if not db_path.exists():
        old = Path.home() / ".forge" / "inference.db"
        if old.exists():
            db_path = old

    if not db_path.exists():
        print("No inference log found. Run 'thermalkit benchmark' first.", flush=True)
        raise typer.Exit(1)

    # Parse --since filter
    since_ts: float | None = None
    if since:
        try:
            since_ts = datetime.strptime(since, "%Y-%m-%d").timestamp()
        except ValueError:
            print(f"Invalid --since date '{since}'. Use YYYY-MM-DD format.", flush=True)
            raise typer.Exit(1)

    conn = sqlite3.connect(str(db_path))
    if since_ts is not None:
        rows = conn.execute(
            "SELECT ts, cpu_temp, gpu_temp, mem_pressure, batt_pct, "
            "compute_unit, batch_size, tokens_generated, tok_sec, reward "
            "FROM calls WHERE ts >= ? ORDER BY ts",
            (since_ts,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts, cpu_temp, gpu_temp, mem_pressure, batt_pct, "
            "compute_unit, batch_size, tokens_generated, tok_sec, reward "
            "FROM calls ORDER BY ts"
        ).fetchall()
    conn.close()

    HEADERS = [
        "ts", "cpu_temp_c", "gpu_temp_c", "mem_pressure_gb", "batt_pct",
        "compute_unit", "batch_size", "tokens_generated", "tok_sec", "reward",
    ]

    def _fmt(val):
        """Format a value for export — empty string for NULL."""
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:.4f}"
        return str(val)

    out_path = Path(output)

    if fmt == "csv":
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            for row in rows:
                writer.writerow([_fmt(v) for v in row])

    elif fmt == "json":
        data = [dict(zip(HEADERS, row)) for row in rows]
        # Replace None with null properly
        out_path.write_text(json.dumps(data, indent=2, default=lambda x: None))

    else:
        print(f"Unknown format '{fmt}'. Use 'csv' or 'json'.", flush=True)
        raise typer.Exit(1)

    filter_str = f" (since {since})" if since else ""
    print(f"Exported {len(rows)} rows{filter_str} → {out_path}", flush=True)
    print(f"Columns: {', '.join(HEADERS)}", flush=True)


if __name__ == "__main__":
    app()
