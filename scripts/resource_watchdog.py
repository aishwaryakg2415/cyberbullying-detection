"""Pause/resume training when the machine is under memory or thermal stress (macOS; partial Linux support).

Every --interval seconds it logs memory/swap/thermal state to results/logs/watchdog.log and:
  * PAUSES (SIGSTOP) SACENet worker processes when
      - macOS memory-pressure level is critical (kern.memorystatus_vm_pressure_level == 4), or
      - free swap < --min-swap-mb while swap is in use, or
      - system available memory < --min-free-pct, or
      - macOS reports thermal throttling (pmset -g therm: CPU_Speed_Limit < 100 or a thermal warning level)
  * RESUMES (SIGCONT) after --recover consecutive healthy checks.
Training is never killed. Temperature in degrees is not readable without sudo on Apple Silicon, so the
thermal signal is macOS's own throttling/warning state.

Usage:  python scripts/resource_watchdog.py &      (stop with Ctrl-C / kill; paused workers are resumed on exit)
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import re
import signal
import subprocess
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "results" / "logs" / "watchdog.log"
WORKER_PATTERNS = ("scripts/run_experiments.py", "scripts/run_explainability.py", "scripts/benchmark_inference.py",
                   "scripts/run_early_detection.py")


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def pressure_level() -> int | None:
    if platform.system() != "Darwin":
        return None
    out = sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]).strip()
    return int(out) if out.isdigit() else None   # 1 normal, 2 warning, 4 critical


def thermal() -> dict:
    if platform.system() != "Darwin":
        return {"throttled": False, "raw": "n/a"}
    out = sh(["pmset", "-g", "therm"])
    lim = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    warn = re.search(r"(thermal|performance) warning level.*?(\d+)", out, re.I)
    speed = int(lim.group(1)) if lim else 100
    recorded_warning = bool(warn) and "No " not in out.split("warning level")[0][-30:]
    return {"cpu_speed_limit": speed, "warning": recorded_warning,
            "throttled": speed < 100 or recorded_warning, "raw": " ".join(out.split())[:160]}


def workers():
    out = []
    for p in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(p.info.get("cmdline") or [])
        if any(w in cmd for w in WORKER_PATTERNS) and p.pid != os.getpid():
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--min-swap-mb", type=int, default=400)
    ap.add_argument("--min-free-pct", type=float, default=4.0)
    ap.add_argument("--recover", type=int, default=3)
    ap.add_argument("--min-disk-gb", type=float, default=5.0)
    args = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    paused: set[int] = set()
    healthy_streak = 0

    def log(msg):
        line = f"{now()} {msg}"
        print(line, flush=True)
        with LOG.open("a") as f:
            f.write(line + "\n")

    def resume_all(reason):
        for pid in list(paused):
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            paused.discard(pid)
        log(f"RESUMED workers ({reason})")

    def on_exit(*_):
        if paused:
            resume_all("watchdog exiting")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, on_exit)
    signal.signal(signal.SIGINT, on_exit)
    log(f"WATCHDOG start interval={args.interval}s min_swap={args.min_swap_mb}MB min_free={args.min_free_pct}%")
    last_status = last_warn = 0.0
    paused_since = 0.0
    while True:
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        lvl = pressure_level()
        th = thermal()
        ws = workers()
        rss = sum(p.memory_info().rss for p in ws if p.is_running()) / 2**30 if ws else 0.0
        swap_free_mb = (sw.total - sw.used) / 2**20
        reasons = []
        if lvl == 4:
            reasons.append("memory pressure CRITICAL")
        # macOS grows swap dynamically while disk space lasts, so low free swap only matters when
        # the disk itself is nearly full (swap cannot grow).
        disk_free_gb = psutil.disk_usage(str(ROOT)).free / 2**30
        if sw.total > 0 and swap_free_mb < args.min_swap_mb and disk_free_gb < args.min_disk_gb:
            reasons.append(f"swap nearly full ({swap_free_mb:.0f} MB free) and disk low ({disk_free_gb:.1f} GB)")
        # NOTE: low "available" RAM alone is not a pause reason: a paused worker keeps its memory,
        # so pausing the main consumer cannot relieve it. macOS's own CRITICAL level and swap
        # exhaustion are used instead (see above); it is logged for information only.
        if th["throttled"]:
            reasons.append(f"thermal throttling (CPU_Speed_Limit={th.get('cpu_speed_limit')})")
        status = (f"mem_avail={vm.available / 2**30:.1f}GB ({vm.available / vm.total:.0%}) pressure={lvl} "
                  f"swap_used={sw.used / 2**30:.2f}/{sw.total / 2**30:.1f}GB workers={len(ws)} rss={rss:.2f}GB "
                  f"thermal={'THROTTLED' if th['throttled'] else 'ok'} disk_free={psutil.disk_usage(str(ROOT)).free / 2**30:.0f}GB")
        mem_only = reasons and all(not r.startswith("thermal") for r in reasons)
        if reasons and mem_only and paused and time.time() - paused_since > 900:
            resume_all(f"memory pause did not relieve pressure in 15 min (workers hold the memory) | {status}")
            paused_since = float("inf")      # do not re-pause for memory until pressure clears once
            reasons = []
        if reasons and mem_only and paused_since == float("inf"):
            reasons = []
        if not reasons and paused_since == float("inf") and lvl != 4:
            paused_since = 0.0
        if reasons:
            healthy_streak = 0
            if not paused:
                paused_since = time.time()
            todo = [p for p in ws if p.pid not in paused]
            for p in todo:
                try:
                    os.kill(p.pid, signal.SIGSTOP)
                    paused.add(p.pid)
                except ProcessLookupError:
                    pass
            if todo:
                log(f"PAUSED {len(todo)} worker(s): {'; '.join(reasons)} | {status}")
            elif time.time() - last_status > 300:
                log(f"STILL-PAUSED: {'; '.join(reasons)} | {status}")
                last_status = time.time()
        else:
            healthy_streak += 1
            if paused and healthy_streak >= args.recover:
                resume_all(f"healthy for {healthy_streak} checks | {status}")
            if time.time() - last_status > 600:   # periodic heartbeat every 10 min
                log(f"OK {status}")
                last_status = time.time()
            elif lvl == 2 and time.time() - last_warn > 300:   # rate-limited
                log(f"WARN memory pressure level=warning | {status}")
                last_warn = time.time()
        paused = {pid for pid in paused if psutil.pid_exists(pid)}
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
