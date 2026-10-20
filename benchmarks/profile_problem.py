"""
Profile one of the benchmark problems with cProfile.

Usage, from this directory::

    python profile_problem.py wind_bubble_1d
    python profile_problem.py electrons_2d --sort cumulative --limit 40

The problem is run once before profiling, so that Numba compilation is excluded.
"""

import argparse
import cProfile
import logging
import pstats

import problems

PROBLEMS = {
    "diffusion_source_1d": problems.diffusion_source_1d,
    "wind_bubble_1d": problems.wind_bubble_1d,
    "full_2d": problems.full_2d,
    "full_2d_dynamic": lambda: problems.full_2d(dynamic=True),
    "electrons_2d": problems.electrons_2d,
    "electrons_2d_dynamic": lambda: problems.electrons_2d(dynamic=True),
    "protons_2d": problems.protons_2d,
    "electron_cooling_1d": problems.electron_cooling_1d,
    "wind_bubble_1d_stress": problems.wind_bubble_1d_stress,
    "full_2d_large": problems.full_2d_large,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("problem", choices=sorted(PROBLEMS))
    parser.add_argument("--sort", default="tottime", help="pstats sort key")
    parser.add_argument("--limit", type=int, default=25, help="rows to print")
    args = parser.parse_args()

    logging.disable(logging.WARNING)
    build = PROBLEMS[args.problem]
    build().run()  # compile the Numba kernels

    problem = build()
    profiler = cProfile.Profile()
    profiler.runcall(problem.run)
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)


if __name__ == "__main__":
    main()
