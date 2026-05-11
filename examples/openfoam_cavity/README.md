# OpenFOAM Cavity Benchmark

This example benchmarks `PODImodels` models on snapshots generated from the
OpenFOAM v2312 lid-driven cavity tutorial.

## Prerequisites

- OpenFOAM v2312 installed and activatable via `of_2312`
- Python environment with project dependencies
- Optional dependency for OpenFOAM I/O:

```bash
uv sync --extra openfoam
```

## Run

From the repository root:

```bash
uv run python examples/openfoam_cavity/benchmark_cavity_models.py --force
```

Useful options:

```bash
uv run python examples/openfoam_cavity/benchmark_cavity_models.py \
  --openfoam-activate of_2312 \
  --lid-velocities 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6 \
  --viscosities 0.0025,0.005,0.01,0.02 \
  --snapshot-mode all \
  --max-snapshots-per-case 12 \
  --rank 5 \
  --ann-epochs 200 \
  --train-ratio 0.75 \
  --seed 42 \
  --force
```

## Outputs

- Console table with per-model:
  - relative Frobenius error
  - fit time
  - predict time
- CSV summary:
  - `examples/openfoam_cavity/results/benchmark_summary.csv`

Generated cases and benchmark outputs are ignored by git:

- `examples/openfoam_cavity/work/`
- `examples/openfoam_cavity/results/`

## Failure diagnostics

- `ModuleNotFoundError: foamToPython`:
  install the optional extra with `uv sync --extra openfoam`.
- `Case source not found`:
  ensure `$FOAM_TUTORIALS` is set by your OpenFOAM environment or that the
  fallback path exists.
- OpenFOAM command failures:
  verify `of_2312` works in your shell and required binaries (`foamCloneCase`,
  `blockMesh`, `icoFoam`) are available.
