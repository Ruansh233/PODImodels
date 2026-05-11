#!/usr/bin/env python3
"""Benchmark PODImodels on OpenFOAM cavity snapshots."""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from PODImodels import (
    PODANN,
    PODGPR,
    PODLinear,
    PODRBF,
    PODRidge,
    PODRidgeGPR,
    PODRidgeRBF,
    fieldsGPR,
    fieldsLinear,
    fieldsRBF,
    fieldsRidge,
    fieldsRidgeGPR,
    fieldsRidgeRBF,
)

DEFAULT_CASE_FALLBACK = Path(
    "/home/ruan/software/OpenFOAM/OpenFOAM-v2312/tutorials/incompressible/icoFoam/cavity/cavity"
)
RESULT_COLUMNS = [
    "model",
    "relative_frobenius_error",
    "fit_seconds",
    "predict_seconds",
    "rank",
    "train_cases",
    "test_cases",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/openfoam_cavity/work"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("examples/openfoam_cavity/results"),
    )
    parser.add_argument("--case-source", type=Path, default=None)
    parser.add_argument("--openfoam-activate", default="of_2312")
    parser.add_argument(
        "--lid-velocities",
        default="0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6",
        help="Comma-separated moving wall velocities.",
    )
    parser.add_argument(
        "--viscosities",
        default="0.0025,0.005,0.01,0.02",
        help="Comma-separated kinematic viscosity (nu) values.",
    )
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--ann-epochs", type=int, default=300)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument(
        "--snapshot-mode",
        choices=["final", "all"],
        default="all",
        help="Use only final snapshot or all positive time snapshots per case.",
    )
    parser.add_argument(
        "--max-snapshots-per-case",
        type=int,
        default=12,
        help="Maximum number of time snapshots to keep per case when snapshot-mode=all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_foam_to_python():
    try:
        return importlib.import_module("foamToPython")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional dependency 'foamToPython'. Install with "
            "`uv sync --extra openfoam`."
        ) from exc


def resolve_case_source(case_source: Path | None) -> Path:
    if case_source is not None:
        return case_source
    tutorials = os.environ.get("FOAM_TUTORIALS")
    if tutorials:
        candidate = Path(tutorials) / "incompressible/icoFoam/cavity/cavity"
        if candidate.exists():
            return candidate
    return DEFAULT_CASE_FALLBACK


def run_openfoam_command(
    command: str,
    activate_cmd: str,
    cwd: Path | None = None,
) -> None:
    full_cmd = f"{activate_cmd} && {command}"
    subprocess.run(
        ["zsh", "-ic", full_cmd],
        cwd=None if cwd is None else str(cwd),
        check=True,
    )


def update_moving_wall_velocity(u_text: str, lid_velocity: float) -> str:
    pattern = re.compile(
        r"(movingWall\s*\{.*?value\s+uniform\s*)\([^)]*\)(\s*;)",
        re.DOTALL,
    )
    replacement = rf"\g<1>({lid_velocity:.8g} 0 0)\g<2>"
    updated, count = pattern.subn(replacement, u_text, count=1)
    if count != 1:
        raise ValueError("Could not find movingWall uniform velocity in 0/U.")
    return updated


def update_kinematic_viscosity(transport_text: str, nu: float) -> str:
    value_pattern = re.compile(r"(^\s*nu\s+)([-+0-9.eE]+)(\s*;)", re.MULTILINE)
    replacement = rf"\g<1>{nu:.8g}\g<3>"
    updated, count = value_pattern.subn(replacement, transport_text, count=1)
    if count == 1:
        return updated

    dimensioned_pattern = re.compile(
        r"(^\s*nu\s*\[[^\]]+\]\s*)([-+0-9.eE]+)(\s*;)", re.MULTILINE
    )
    updated, count = dimensioned_pattern.subn(replacement, transport_text, count=1)
    if count == 1:
        return updated
    raise ValueError("Could not find viscosity entry for 'nu ...;' in transportProperties.")


def prepare_case_for_parameters(
    case_source: Path,
    case_dir: Path,
    lid_velocity: float,
    viscosity: float,
    activate_cmd: str,
    force: bool,
) -> None:
    if case_dir.exists():
        if not force:
            raise FileExistsError(
                f"Case directory exists: {case_dir}. Use --force to overwrite."
            )
        shutil.rmtree(case_dir)
    case_dir.parent.mkdir(parents=True, exist_ok=True)
    run_openfoam_command(
        f"foamCloneCase {case_source} {case_dir}",
        activate_cmd=activate_cmd,
    )

    u_file = case_dir / "0/U"
    updated = update_moving_wall_velocity(u_file.read_text(), lid_velocity)
    u_file.write_text(updated)
    transport_file = case_dir / "constant/transportProperties"
    updated_transport = update_kinematic_viscosity(
        transport_file.read_text(),
        viscosity,
    )
    transport_file.write_text(updated_transport)

    run_openfoam_command("blockMesh", activate_cmd=activate_cmd, cwd=case_dir)
    run_openfoam_command("icoFoam", activate_cmd=activate_cmd, cwd=case_dir)


def find_latest_time_dir(case_dir: Path) -> Path:
    candidates = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            tval = float(entry.name)
        except ValueError:
            continue
        if tval > 0:
            candidates.append((tval, entry))
    if not candidates:
        raise FileNotFoundError(f"No positive numeric time directories in {case_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def list_positive_time_dirs(case_dir: Path) -> List[Tuple[float, Path]]:
    candidates = []
    for entry in case_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            tval = float(entry.name)
        except ValueError:
            continue
        if tval > 0:
            candidates.append((tval, entry))
    if not candidates:
        raise FileNotFoundError(f"No positive numeric time directories in {case_dir}")
    candidates.sort(key=lambda item: item[0])
    return candidates


def flatten_velocity_snapshot(field_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(field_array)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr.reshape(-1)
    return arr.reshape(-1)


def extract_final_u_snapshot(case_dir: Path, foam_module) -> np.ndarray:
    latest_time = find_latest_time_dir(case_dir)
    u_path = latest_time / "U"
    field = foam_module.OFField(str(u_path), "vector", read_data=True)
    return flatten_velocity_snapshot(np.asarray(field.internalField))


def select_time_dirs(
    case_dir: Path,
    snapshot_mode: str,
    max_snapshots_per_case: int,
) -> List[Tuple[float, Path]]:
    if snapshot_mode == "final":
        latest = find_latest_time_dir(case_dir)
        return [(float(latest.name), latest)]
    times = list_positive_time_dirs(case_dir)
    if max_snapshots_per_case <= 0 or len(times) <= max_snapshots_per_case:
        return times
    keep = np.linspace(0, len(times) - 1, num=max_snapshots_per_case, dtype=int)
    keep = np.unique(keep)
    return [times[i] for i in keep]


def extract_u_snapshots(
    case_dir: Path,
    foam_module,
    snapshot_mode: str,
    max_snapshots_per_case: int,
) -> List[Tuple[float, np.ndarray]]:
    out: List[Tuple[float, np.ndarray]] = []
    for tval, time_dir in select_time_dirs(
        case_dir=case_dir,
        snapshot_mode=snapshot_mode,
        max_snapshots_per_case=max_snapshots_per_case,
    ):
        u_path = time_dir / "U"
        field = foam_module.OFField(str(u_path), "vector", read_data=True)
        out.append((tval, flatten_velocity_snapshot(np.asarray(field.internalField))))
    return out


def deterministic_split_indices(
    n_samples: int, train_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to split train/test.")
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_train = int(round(n_samples * train_ratio))
    n_train = max(1, min(n_samples - 1, n_train))
    return indices[:n_train], indices[n_train:]


def relative_frobenius(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.linalg.norm(y_true))
    if denom == 0.0:
        return float(np.linalg.norm(y_true - y_pred))
    return float(np.linalg.norm(y_true - y_pred) / denom)


def model_builders(rank: int, ann_epochs: int) -> Dict[str, Callable[[], object]]:
    return {
        "fieldsLinear": lambda: fieldsLinear(),
        "fieldsRidge": lambda: fieldsRidge(),
        "fieldsGPR": lambda: fieldsGPR(),
        "fieldsRidgeGPR": lambda: fieldsRidgeGPR(),
        "fieldsRBF": lambda: fieldsRBF(kernel="linear"),
        "fieldsRidgeRBF": lambda: fieldsRidgeRBF(kernel="linear"),
        "PODLinear": lambda: PODLinear(rank=rank),
        "PODRidge": lambda: PODRidge(rank=rank),
        "PODRBF": lambda: PODRBF(rank=rank, kernel="linear"),
        "PODRidgeRBF": lambda: PODRidgeRBF(rank=rank, kernel="linear"),
        "PODGPR": lambda: PODGPR(rank=rank),
        "PODRidgeGPR": lambda: PODRidgeGPR(rank=rank),
        "PODANN": lambda: PODANN(
            rank=rank,
            num_epochs=ann_epochs,
            hidden_layer_sizes=[32, 16],
        ),
    }


def benchmark_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    rank: int,
    ann_epochs: int,
    builders: Dict[str, Callable[[], object]] | None = None,
) -> List[dict]:
    rows: List[dict] = []
    chosen_builders = builders if builders is not None else model_builders(rank, ann_epochs)
    for name, build in chosen_builders.items():
        model = build()
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = model.predict(x_test)
        pred_s = time.perf_counter() - t1

        err = relative_frobenius(y_test, y_pred)
        rows.append(
            {
                "model": name,
                "relative_frobenius_error": err,
                "fit_seconds": fit_s,
                "predict_seconds": pred_s,
                "rank": int(rank),
                "train_cases": int(x_train.shape[0]),
                "test_cases": int(x_test.shape[0]),
            }
        )
    return rows


def write_results(rows: Iterable[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_results(rows: List[dict]) -> None:
    header = (
        f"{'model':<12} {'rel_F':>12} {'fit_s':>12} {'pred_s':>12} "
        f"{'rank':>6} {'train':>8} {'test':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<12} "
            f"{row['relative_frobenius_error']:>12.6e} "
            f"{row['fit_seconds']:>12.4f} "
            f"{row['predict_seconds']:>12.4f} "
            f"{row['rank']:>6} "
            f"{row['train_cases']:>8} "
            f"{row['test_cases']:>8}"
        )


def parse_lid_velocities(raw: str) -> List[float]:
    vals = [item.strip() for item in raw.split(",") if item.strip()]
    if not vals:
        raise ValueError("No lid velocities provided.")
    return [float(v) for v in vals]


def parse_viscosities(raw: str) -> List[float]:
    vals = [item.strip() for item in raw.split(",") if item.strip()]
    if not vals:
        raise ValueError("No viscosities provided.")
    return [float(v) for v in vals]


def main() -> None:
    args = parse_args()
    foam_module = load_foam_to_python()
    case_source = resolve_case_source(args.case_source)
    if not case_source.exists():
        raise FileNotFoundError(f"Case source not found: {case_source}")

    lid_velocities = parse_lid_velocities(args.lid_velocities)
    viscosities = parse_viscosities(args.viscosities)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    x_rows = []
    y_rows = []
    for lid_velocity, viscosity in itertools.product(lid_velocities, viscosities):
        case_dir = args.work_dir / f"cavity_u_{lid_velocity:.8g}_nu_{viscosity:.8g}"
        prepare_case_for_parameters(
            case_source=case_source,
            case_dir=case_dir,
            lid_velocity=lid_velocity,
            viscosity=viscosity,
            activate_cmd=args.openfoam_activate,
            force=args.force,
        )
        snapshots = extract_u_snapshots(
            case_dir=case_dir,
            foam_module=foam_module,
            snapshot_mode=args.snapshot_mode,
            max_snapshots_per_case=args.max_snapshots_per_case,
        )
        for time_value, snapshot in snapshots:
            x_rows.append([lid_velocity, viscosity, time_value])
            y_rows.append(snapshot)

    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)

    train_idx, test_idx = deterministic_split_indices(
        n_samples=x.shape[0], train_ratio=args.train_ratio, seed=args.seed
    )
    rows = benchmark_models(
        x_train=x[train_idx],
        y_train=y[train_idx],
        x_test=x[test_idx],
        y_test=y[test_idx],
        rank=args.rank,
        ann_epochs=args.ann_epochs,
    )

    csv_path = args.results_dir / "benchmark_summary.csv"
    write_results(rows, csv_path)
    print_results(rows)
    print(f"\nSaved benchmark summary to: {csv_path}")


if __name__ == "__main__":
    main()
