"""Runs the full experiment grid

Usage:
  python run_experiments.py                          # resnet50, 3 seeds, both split modes
  python run_experiments.py --backbone tiny --max-epochs 3 --workers 0 --no-pretrained   # smoke test
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-epochs", default="30")
    ap.add_argument("--epochs-head", default="4")
    ap.add_argument("--workers", default="2")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    for mode in ("cluster", "random"):
        for seed in range(args.seeds):
            if Path(f"runs/{mode}_s{seed}/metrics.json").exists():  # finished before a Colab disconnect
                print(f"skip {mode}_s{seed}: already finished")
                continue
            cmd =["05_train.py", "--name", f"{mode}_s{seed}", "--seed", str(seed), "--split-mode", mode,
                   "--backbone", args.backbone, "--max-epochs", args.max_epochs,
                   "--epochs-head", args.epochs_head, "--workers", args.workers]
            if args.no_pretrained:
                cmd.append("--no-pretrained")
            run(*cmd)
    run("", "--runs", *[f"runs/cluster_s{s}" for s in range(args.seeds)])
    run("")


if __name__ == "__main__":
    main()
