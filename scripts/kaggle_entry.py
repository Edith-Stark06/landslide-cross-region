"""Kaggle kernel entry point: runs the jobs listed in the code dataset's jobs.json on the kernel's GPU.

Inputs (attached as private datasets): landslide-xr-code (src/, splits/, configs/, jobs.json, GIT_HASH) and
landslide-xr-<dataset> for every dataset the jobs use. Outputs go to /kaggle/working/results (+ results.zip).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")


def find_dataset(slug: str) -> Path:
    for p in INPUT.rglob("*"):
        if p.is_dir() and p.name == slug:
            return p
    raise FileNotFoundError(f"dataset {slug} is not attached to this kernel")


def stage(msg):
    print(f"[entry {time.strftime('%H:%M:%S')}] {msg}", flush=True)


stage("start")
subprocess.run(["nvidia-smi", "-L"], check=False, timeout=60)
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True, timeout=600)
except subprocess.TimeoutExpired:
    stage("pip install timed out (internet unavailable?)")
    raise
stage("pip done")

code = WORK / "code"
if code.exists():
    shutil.rmtree(code)
shutil.copytree(find_dataset("landslide-xr-code"), code)
JOBS_FILE = "jobs.json"   # make_kernel.ps1 -JobsName rewrites this line so several kernels can share one code dataset
jobs = json.loads((code / JOBS_FILE).read_text())

proc = WORK / "processed"
proc.mkdir(exist_ok=True)
for d in INPUT.rglob("meta.csv"):                       # dataset dirs hold images.npy, masks.npy, meta.csv, stats.json
    slug = d.parent.name
    if slug.startswith("landslide-xr-") and not (proc / slug.removeprefix("landslide-xr-")).exists():
        (proc / slug.removeprefix("landslide-xr-")).symlink_to(d.parent)
print("datasets:", sorted(p.name for p in proc.iterdir()), flush=True)
if not any(proc.iterdir()):                       # fail fast and show what Kaggle actually mounted
    for p in sorted(INPUT.glob("*"))[:30]:
        print("  /kaggle/input/" + p.name, [c.name for c in sorted(p.glob("*"))[:8]], flush=True)
    sys.exit("no landslide-xr-* data datasets are mounted")

env = dict(os.environ, LANDSLIDE_PROCESSED=str(proc), PYTHONPATH=str(code / "src"),
           GIT_HASH=(code / "GIT_HASH").read_text().strip() if (code / "GIT_HASH").exists() else "unknown")
out = WORK / "results" / "runs"
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for job in jobs:
    for seed in job["seeds"]:
        print(f"=== {job['config']} seed {seed} (elapsed {time.time() - t0:.0f}s) ===", flush=True)
        try:
            r = subprocess.run([sys.executable, "-u", "-m", "landslide.run", "--config", str(code / job["config"]), "--seed", str(seed),
                                "--out", str(out)], env=env, cwd=code, timeout=job.get("timeout_s", 1800))
            if r.returncode != 0:
                print(f"!!! job failed: {job['config']} seed {seed} (exit {r.returncode}); continuing", flush=True)
        except subprocess.TimeoutExpired:
            print(f"!!! job TIMED OUT: {job['config']} seed {seed}", flush=True)
shutil.make_archive(str(WORK / "results"), "zip", WORK / "results")
print("done; total seconds", time.time() - t0)
