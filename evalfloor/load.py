"""Get scores out of whatever your tuning loop left on disk.

`check()` needs two things: every variant's score, and how many examples
each was scored on. Collecting the first by hand is the step where people
give up, so these read it from the shapes tuning tools already produce.

No hard dependency on any of them. An Optuna study is read by attribute, a
CSV by the standard library, so importing this module still pulls in
nothing.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable


def from_records(records: Iterable[Any], key: str) -> list[float]:
    """Scores out of dicts, objects, or rows, by field name.

    Accepts a mapping key, an attribute, or an index into a sequence, so the
    same call works on a list of dicts, a list of dataclasses, and a list of
    tuples. Rows whose field is missing or not a number are skipped rather
    than crashing a run someone is part-way through: a sweep with two failed
    trials should still be checkable.
    """
    out = []
    for r in records:
        v = None
        if isinstance(r, dict):
            v = r.get(key)
        elif hasattr(r, key):
            v = getattr(r, key)
        elif isinstance(r, (list, tuple)):
            try:
                v = r[int(key)]
            except (ValueError, IndexError):
                v = None
        if isinstance(v, bool) or v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def from_optuna(study, attr: str = "value") -> list[float]:
    """Completed trial values from an Optuna study.

    Takes the study object, not a storage URL, so this never imports optuna.
    Pruned and failed trials are dropped: they were never candidates for the
    maximum, and counting them would inflate k and so the floor.
    """
    trials = getattr(study, "trials", study)
    vals = []
    for t in trials:
        state = getattr(t, "state", None)
        if state is not None and getattr(state, "name", str(state)).upper().find("COMPLETE") < 0:
            continue
        v = getattr(t, attr, None)
        if v is not None:
            vals.append(float(v))
    return vals


def from_csv(path: str, column: str) -> list[float]:
    """One column of a CSV or TSV, picked by header name."""
    delim = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter=delim))
    if rows and column not in rows[0]:
        raise KeyError(f"no column {column!r}; found {list(rows[0])}")
    return from_records(rows, column)


def from_jsonl(path: str, key: str) -> list[float]:
    """One field out of a JSONL file, one object per line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return from_records(rows, key)


def from_json(path: str, key: str) -> list[float]:
    """One field out of a JSON array, or a bare array of numbers."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        d = d.get("trials") or d.get("runs") or d.get("results") or d.get("scores") or []
    if d and isinstance(d[0], (int, float)):
        return [float(x) for x in d]
    return from_records(d, key)


def load(path: str, key: str) -> list[float]:
    """Read scores from a file, picking the reader by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv", ".tab"):
        return from_csv(path, key)
    if ext in (".jsonl", ".ndjson"):
        return from_jsonl(path, key)
    if ext == ".json":
        return from_json(path, key)
    raise ValueError(f"unsupported extension {ext!r}; use .csv, .tsv, .jsonl or .json")
