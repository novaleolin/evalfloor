"""overtuned -- your eval score went up; did anything actually get better?

    check(scores, n_examples)          did the search prove anything?
    confirm(baseline_hits, new_hits)   does the winner survive held-out data?

Both are arithmetic on numbers your existing tuning loop already produced:
no model, no API key, no dependencies. The schema optimiser in `optimize`
is the same machinery applied to itself, and needs the extras.
"""
from .check import (Check, Confirm, StagedFloor, check, confirm, eb_shrink,
                    selection_floor, staged_floor)

__version__ = "0.1.0"
__all__ = ["check", "confirm", "Check", "Confirm", "selection_floor",
           "eb_shrink", "staged_floor", "StagedFloor"]


def __getattr__(name):  # noqa: D401
    """Expose the optimiser lazily, so `import overtuned` stays dependency-free.

    check() and confirm() are the reason most people arrive, and requiring
    torch to compute a binomial expectation would be absurd. The optimiser
    imports its own dependencies only when something actually reaches for it.
    """
    where = {"optimize": ("optimize", "optimize"), "Result": ("optimize", "Result"),
             "Schema": ("schema", "Schema"), "Task": ("task", "Task"),
             "Example": ("task", "Example"), "from_jsonl": ("task", "from_jsonl"),
             "LocalBackend": ("backends", "LocalBackend"),
             "JevBackend": ("backends", "JevBackend")}
    if name not in where:
        raise AttributeError(name)
    # import_module, not `from . import x`: the submodule `optimize` and the
    # function `optimize` share a name, so the plain form re-enters this
    # very function and recurses until the stack gives out.
    import importlib
    mod, attr = where[name]
    return getattr(importlib.import_module(f".{mod}", __name__), attr)
