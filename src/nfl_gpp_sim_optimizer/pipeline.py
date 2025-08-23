import argparse
from .pipeline import compare_apply, build_lineups

def main():
    parser = argparse.ArgumentParser(prog="nfl-gpp")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("compare-apply")
    p1.add_argument("--players", required=True)
    p1.add_argument("--sims")
    p1.add_argument("--defcsv")
    p1.add_argument("--corrcsv")
    p1.add_argument("--outdir", default="out")

    p2 = sub.add_parser("build-lineups")
    p2.add_argument("--players", required=True)
    p2.add_argument("--corrcsv")
    p2.add_argument("--preset", choices=["se","mid","large"], default="large")
    p2.add_argument("--n", type=int, default=150)
    p2.add_argument("--outdir", default="out")

    args = parser.parse_args()
    if args.cmd == "compare-apply":
        compare_apply(args)
    elif args.cmd == "build-lineups":
        build_lineups(args)
