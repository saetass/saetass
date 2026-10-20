"""
Print the median of each benchmark in several saved runs and the speedup over the first.

Usage, from the repository root::

    python benchmarks/compare.py 0001 0003
    python benchmarks/compare.py baseline baseline-serial

Each argument selects the saved run in ``.benchmarks/`` whose file name contains it.
"""

import argparse
import json
from pathlib import Path

STORAGE = Path(__file__).resolve().parent.parent / ".benchmarks"


def _find(run: str) -> Path:
    # An exact name (after the run number) wins over names that merely contain it
    matches = sorted(STORAGE.glob(f"*/[0-9][0-9][0-9][0-9]_{run}.json"))
    if not matches:
        matches = sorted(STORAGE.glob(f"*/*{run}*.json"))
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise SystemExit(
            f"{run!r} must match one saved run in {STORAGE}, found {found}"
        )
    return matches[0]


def _medians(path: Path) -> dict[str, tuple[float, int | None]]:
    data = json.loads(path.read_text())
    return {
        bench["name"]: (bench["stats"]["median"], bench["extra_info"].get("cells"))
        for bench in data["benchmarks"]
    }


def _format_time(seconds: float) -> str:
    if seconds >= 1e-1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-4:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.2f} us"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "runs", nargs="+", help="saved runs, the first is the reference"
    )
    args = parser.parse_args()

    paths = [_find(run) for run in args.runs]
    medians = [_medians(path) for path in paths]
    names = list(medians[0])
    for run in medians[1:]:
        names += [name for name in run if name not in names]

    width = max(len(name) for name in names)
    header = f"{'benchmark':<{width}}  " + "  ".join(
        f"{path.stem[:24]:>24}" for path in paths
    )
    print(header)
    print("-" * len(header))
    for name in names:
        reference = medians[0].get(name)
        cells = []
        for run in medians:
            if name not in run:
                cells.append(f"{'-':>24}")
                continue
            median, n_cells = run[name]
            text = _format_time(median)
            if n_cells:
                text += f" {median / n_cells * 1e9:5.1f}ns"
            if reference is not None and run is not medians[0]:
                text += f" x{reference[0] / median:5.1f}"
            cells.append(f"{text:>24}")
        print(f"{name:<{width}}  " + "  ".join(cells))
    print("\nns: median time per cell; x: speedup over the first run")


if __name__ == "__main__":
    main()
