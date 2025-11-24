#!/usr/bin/env python3

# python wait_then_run.py --gpus 8 \
#   /workspace/mnt/qqt/project/SparseAttn/sparseattn/run_scripts/prulong_masksonly_qwen_debug.sh \
#   /workspace/mnt/qqt/project/SparseAttn/sparseattn/run_scripts/next_experiment.sh

import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from mptools import mpenv


def main():
    parser = argparse.ArgumentParser(
        description="Wait for idle GPUs, then sequentially execute multiple bash scripts and save logs."
    )
    parser.add_argument(
        "--gpus", type=int, default=8, help="Number of idle GPUs to wait for (default: 1)"
    )
    parser.add_argument(
        "scripts", nargs="+", help="Paths to bash scripts to execute sequentially"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory to save log files (default: ./logs)",
    )

    args = parser.parse_args()

    # === Step 1: Wait for GPUs ===
    print(f"[INFO] Waiting for {args.gpus} idle GPU(s)...")
    mpenv.wait_for_idle_gpus(args.gpus)
    print(f"[INFO] ✅ GPUs are now idle. Starting scripts...\n")

    # === Step 2: Prepare log directory ===
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # === Step 3: Execute scripts sequentially ===
    for script_path in args.scripts:
        script = Path(script_path)
        if not script.exists():
            print(f"[ERROR] Script not found: {script}")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{script.stem}_{timestamp}.log"

        print(f"[INFO] Running: {script}")
        print(f"[INFO] Logging to: {log_file}\n")

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                ["bash", str(script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in process.stdout:
                print(line, end="")
                f.write(line)
            process.wait()

        if process.returncode == 0:
            print(f"\n[INFO] ✅ Finished successfully: {script}\n")
        else:
            print(f"\n[ERROR] ❌ Script failed ({script}), return code {process.returncode}\n")

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
