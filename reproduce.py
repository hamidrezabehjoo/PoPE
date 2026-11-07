#!/usr/bin/env python3
"""Single entry point for figure building and numerical regeneration."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    figures = subparsers.add_parser(
        "figures", help="build manuscript figures from a numerical data directory")
    figures.add_argument("--only", default="all",
                         help="all or comma-separated: gaussian,early,ess,dimension,gmm,densities")
    figures.add_argument("--data-dir", type=Path, default=ROOT / "data")
    figures.add_argument("--output-dir", type=Path, default=ROOT / "figures")

    simulate = subparsers.add_parser(
        "simulate", help="regenerate numerical data without overwriting data/")
    simulate.add_argument("suite", choices=("gaussian", "thresholds", "gmm",
                                            "gmm-moments", "comparisons"))
    simulate.add_argument("--quick", action="store_true",
                          help="small test settings; not the paper's numerical accuracy")
    simulate.add_argument("--output-dir", type=Path,
                          help="default: results/<suite>")

    args = parser.parse_args()
    if args.command == "figures":
        from pope.figures import build_figures
        build_figures(args.data_dir, args.output_dir, args.only)
        return

    output = args.output_dir or ROOT / "results" / args.suite
    if args.suite == "gaussian":
        from pope.experiments import run_gaussian
        run_gaussian(output, quick=args.quick)
    elif args.suite == "thresholds":
        from pope.experiments import run_gaussian_thresholds
        run_gaussian_thresholds(output, quick=args.quick)
    elif args.suite == "gmm":
        from pope.experiments import run_gmm
        run_gmm(output, quick=args.quick)
    elif args.suite == "gmm-moments":
        from pope.experiments import run_gmm_moments
        run_gmm_moments(output, quick=args.quick)
    else:
        from pope.diagnostics import run_comparisons
        run_comparisons(output, quick=args.quick)
    print(f"Numerical data written to {output.resolve()}")


if __name__ == "__main__":
    main()
