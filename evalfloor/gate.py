"""Fail a build when a reported gain is inside the selection floor.

Printing a floor to a terminal only helps the person reading the terminal.
The same number is more useful wired into the pipeline that produced the
scores, where it can stop a tuning result from being promoted on noise.

    python -m evalfloor.gate --scores 0.62 0.65 0.69 --n 200
    python -m evalfloor.gate sweep.csv --score-col accuracy --n 200
    python -m evalfloor.gate results.json

Exit codes are the interface: 0 when the gain clears the floor, 1 when it
does not, 2 on bad input. Everything else goes to stderr so stdout can be
piped, and `--json` makes stdout a single machine-readable object.
"""
from __future__ import annotations

import argparse
import json
import sys

from .check import check, confirm
from .load import load


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m evalfloor.gate",
        description="Exit non-zero when a tuning gain is inside the floor.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("path", nargs="?",
                     help="a .csv/.tsv/.jsonl/.json of your run (use "
                          "--score-col to name the column), or a JSON object "
                          "with scores, n_examples and optionally "
                          "baseline_hits and winner_hits")
    src.add_argument("--scores", nargs="+", type=float,
                     help="every variant's score, winner included")
    ap.add_argument("--n", "--n-examples", dest="n", type=int,
                    help="examples each variant was scored on")
    ap.add_argument("--baseline", type=float, default=None,
                    help="baseline score (default: the first one)")
    ap.add_argument("--score-col", "--score-key", dest="col", default=None,
                    help="column or field holding each variant's score")
    ap.add_argument("--json", action="store_true",
                    help="emit one JSON object on stdout")
    a = ap.parse_args(argv)

    if a.path and a.col:
        # A results file from a sweep: one row per variant, one column of
        # scores. This is the shape people already have, and collecting it
        # into a list by hand is where they stop bothering.
        scores, n, base, pair = load(a.path, a.col), a.n, a.baseline, (None, None)
    elif a.path:
        d = _load(a.path)
        scores, n = d.get("scores"), d.get("n_examples")
        base = d.get("baseline")
        pair = (d.get("baseline_hits"), d.get("winner_hits"))
    else:
        scores, n, base, pair = a.scores, a.n, a.baseline, (None, None)

    if not scores or not n:
        print("need scores and n_examples (pass --n with a results file)",
              file=sys.stderr)
        return 2

    c = check(scores, n, base)
    out = {"n_candidates": c.n_candidates, "n_examples": c.n_examples,
           "baseline": c.baseline, "best": c.best,
           "apparent_gain": c.apparent_gain, "floor": c.floor,
           "shrunk": c.shrunk, "beats_floor": c.beats_floor}

    ok = c.beats_floor
    if pair[0] is not None and pair[1] is not None:
        cf = confirm(pair[0], pair[1])
        out.update(wins=cf.wins, losses=cf.losses, p_value=cf.p_value,
                   confirmed=cf.confirmed, underpowered=cf.underpowered)
        # With held-out data available it, not the floor, is the criterion:
        # the floor governs the split the search ran on, and a held-out
        # result was never selected on.
        ok = cf.confirmed

    if a.json:
        print(json.dumps(out))
    else:
        print(c, file=sys.stderr)
        if "confirmed" in out:
            print(confirm(pair[0], pair[1]), file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
