"""Measure the deployment side of the accuracy trade.

The tabular benchmark quantifies what the engine gives up in accuracy. This
quantifies what it buys back, so "a far lighter deployment story" stops being
an assertion. Three measurements, no estimates:

1. **Artifact size** — bytes on disk of an exported artifact folder.
2. **Inference dependency closure** — the wheels a fresh inference environment
   would download, resolved with `pip install --dry-run --ignore-installed`
   and sized from Content-Length. Reported for the engine artifact's own
   pinned requirements and for AutoGluon at three operating points, including
   the extras set `benchmarks/tabular/requirements.txt` actually installs.
3. **Scoring latency** — p50/p95 of `POST /predict` against `serving/app.py`
   under uvicorn over real HTTP (not TestClient), after warmup.

Not measured, and therefore not claimed: container image size (needs a Docker
daemon) and AutoGluon's scoring latency (needs a trained AutoGluon model).

    python measure.py --artifact /path/to/exported/artifact

Writes results/results.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
import pandas as pd

HERE = Path(__file__).parent
SERVING = HERE.parent.parent / "serving"
PORT = 8077

AUTOGLUON_SPECS = {
    "autogluon_base": ["autogluon.tabular"],
    "autogluon_gbm": ["autogluon.tabular[lightgbm,xgboost,catboost]>=1.5"],
    # exactly what benchmarks/tabular/requirements.txt installs
    "autogluon_benchmark_env": ["autogluon.tabular[lightgbm,xgboost,catboost,fastai]>=1.5"],
}


# ------------------------------------------------------------------ closures
def _wheel_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def closure(specs: list, tag: str) -> dict | None:
    """Full download closure for `specs`, independent of what is installed here."""
    report = HERE / "results" / f"_closure_{tag}.json"
    cmd = ["pip", "install", "--dry-run", "--ignore-installed", "--quiet",
           "--report", str(report), *specs]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(f"  closure({tag}) failed: {r.stderr.strip()[:300]}")
        return None
    pkgs = json.loads(report.read_text())["install"]
    sized = [(p["metadata"]["name"], p["metadata"]["version"],
              _wheel_size(p.get("download_info", {}).get("url", "")))
             for p in pkgs]
    report.unlink(missing_ok=True)
    return {"n_packages": len(sized), "total_bytes": sum(s[2] for s in sized),
            "largest": sorted(sized, key=lambda s: -s[2])[:8]}


# ------------------------------------------------------------------- latency
def _percentiles(xs: list) -> dict:
    xs = sorted(xs)
    return {"p50_ms": round(statistics.median(xs) * 1000, 2),
            "p95_ms": round(xs[int(0.95 * (len(xs) - 1))] * 1000, 2),
            "mean_ms": round(statistics.fmean(xs) * 1000, 2), "n_requests": len(xs)}


def latency(artifact: Path, rows_source: Path) -> dict:
    meta = json.loads((artifact / "metadata.json").read_text())
    raw = pd.read_parquet(rows_source).drop(columns=[meta["target"]], errors="ignore")
    rows_all = raw.head(1000).to_dict(orient="records")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "warning"],
        cwd=str(SERVING), env={**os.environ, "MODEL_DIR": str(artifact)},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base, out = f"http://127.0.0.1:{PORT}", {}
    try:
        with httpx.Client() as c:
            deadline = time.time() + 180
            while time.time() < deadline:
                try:
                    if c.get(base + "/health", timeout=5).status_code == 200:
                        break
                except Exception:
                    time.sleep(0.5)
            else:
                raise RuntimeError("service never became ready")

            for label, n, reps in [("single_row", 1, 200), ("batch_1000", 1000, 40)]:
                body = {"rows": rows_all[:n]}
                for _ in range(5):
                    c.post(base + "/predict", json=body, timeout=120)
                lat = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    c.post(base + "/predict", json=body, timeout=120).raise_for_status()
                    lat.append(time.perf_counter() - t0)
                out[label] = {**_percentiles(lat), "rows_per_request": n}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, type=Path)
    ap.add_argument("--rows", type=Path,
                    default=HERE.parent / "tabular" / "results" / "_diamonds.parquet")
    ap.add_argument("--skip-closures", action="store_true")
    args = ap.parse_args()

    (HERE / "results").mkdir(exist_ok=True)
    art = args.artifact
    files = {f.name: f.stat().st_size for f in art.iterdir() if f.is_file()}
    # requirements.txt carries a header comment and commented-out optional
    # extras — parse it as pip would, not by splitting on whitespace
    pins = [line.split("#")[0].strip()
            for line in (art / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]
    pins = [p for p in pins if p]

    out = {"artifact": {"total_bytes": sum(files.values()),
                        "files": dict(sorted(files.items(), key=lambda kv: -kv[1])),
                        "pinned_requirements": pins,
                        "champion": json.loads((art / "metadata.json").read_text())["champion"]},
           "latency": latency(art, args.rows)}

    # The floor: the four libraries every artifact pins regardless of champion.
    # A pure-sklearn champion needs exactly these; the gap to the as-pinned row
    # is what pinning the whole available GBM roster costs.
    core = [p for p in pins if p.split("==")[0] in
            ("scikit-learn", "pandas", "numpy", "joblib")]

    if not args.skip_closures:
        out["closures"] = {}
        for tag, specs in {"engine_artifact_as_pinned": pins,
                           "engine_artifact_sklearn_only_champion": core,
                           **AUTOGLUON_SPECS}.items():
            print(f"  resolving closure: {tag} ...")
            c = closure(specs, tag)
            if c:
                out["closures"][tag] = c

    (HERE / "results" / "results.json").write_text(json.dumps(out, indent=2))
    print(f"\n  artifact: {out['artifact']['total_bytes'] / 1e6:.2f} MB")
    for k, v in out["latency"].items():
        print(f"  {k:<12} p50 {v['p50_ms']:.2f} ms   p95 {v['p95_ms']:.2f} ms")
    for tag, c in out.get("closures", {}).items():
        print(f"  {tag:<28} {c['n_packages']:>4} pkgs  {c['total_bytes'] / 1e6:>7.0f} MB")


if __name__ == "__main__":
    main()
