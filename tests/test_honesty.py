"""The claims this library makes about itself, as tests.

Each one is written as an attack: hand the machinery the exact situation it
exists to catch, and require it to catch it.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from overtuned.optimize import Result, eb_shrink, selection_floor
from overtuned.schema import Schema, mutate_threshold
from overtuned.task import Example, Task, contains, exact, f1, predict


def test_selection_floor_grows_with_candidates():
    """More candidates, more free gain -- that is the whole warning."""
    floors = [selection_floor(k, 100, 0.5, reps=600, seed=1)
              for k in (1, 5, 20, 100)]
    assert floors[0] < 0.02, floors
    assert floors == sorted(floors), floors
    assert floors[-1] > 0.08, floors


def test_selection_floor_shrinks_with_more_examples():
    """Evaluating on more data buys back what selection took."""
    small = selection_floor(30, 50, 0.5, reps=600, seed=1)
    big = selection_floor(30, 2000, 0.5, reps=600, seed=1)
    assert big < small / 2, (small, big)


def test_eb_pulls_a_pure_noise_winner_back():
    rng = random.Random(0)
    scores = [rng.gauss(0.5, 0.05) for _ in range(40)]      # all equally good
    assert max(scores) - 0.5 > 0.05
    assert abs(eb_shrink(scores, 100) - 0.5) < 0.05


def test_eb_leaves_a_real_winner_alone():
    """Shrinkage must not flatten a candidate that is genuinely ahead."""
    scores = [0.50] * 20 + [0.80]
    assert eb_shrink(scores, 400) > 0.70, eb_shrink(scores, 400)


def test_a_train_only_gain_is_not_credible():
    """The failure this tool exists to prevent: quoting the training score."""
    r = Result(best=None, baseline_train=0.40, reported=0.62,
               heldout_baseline=0.40, heldout_best=0.40,
               null_gain=0.20, shrunk=0.41, wins=3, losses=3, p_value=1.0)
    assert not r.credible


def test_the_floor_has_no_claim_on_held_out_data():
    """A held-out gain stands on its own, even below the training floor.

    An earlier version required the TRAIN gain to clear the floor and so
    rejected a run whose held-out gain was more than twice its train gain.
    The floor discounts the score the search selected on; the held-out score
    was never selected on.
    """
    r = Result(best=None, baseline_train=0.44, reported=0.52,
               heldout_baseline=0.33, heldout_best=0.54,
               null_gain=0.14, shrunk=0.45, wins=13, losses=3, p_value=0.02)
    assert r.real_gain > 0.2
    assert r.reported - r.baseline_train < r.null_gain
    assert r.credible


def test_metrics_disagree_the_way_they_should():
    assert exact({"a"}, {"a"}) == 1.0 and exact({"a", "b"}, {"a"}) == 0.0
    assert contains({"a", "b"}, {"a"}) == 1.0      # over-calling is free
    assert 0.6 < f1({"a", "b"}, {"a"}) < 0.7


def test_threshold_decides_what_a_choice_returns():
    """A choice threshold is RELATIVE to the argmax, not absolute.

    Worth stating because it is easy to mis-read: at 0.9 with an argmax of
    0.50, a rival on 0.47 is kept, since 0.47 >= 0.9 * 0.50. Excluding it
    takes a threshold above 0.94. Absolute thinking here silently changes
    how many labels a schema emits.
    """
    s = Schema(questions={"q": {"type": "choice", "criteria": {"x": "", "y": ""}}},
               thresholds={"q": 0.96})
    ans = {"q": {"probabilities": {"x": 0.5, "y": 0.47}, "choice": "x"}}
    assert predict(ans, s) == {"x"}
    s.thresholds["q"] = 0.9
    assert predict(ans, s) == {"x", "y"}


def test_split_is_deterministic_and_disjoint():
    t = Task([Example({"t": str(i)}, {"a"}) for i in range(40)], exact)
    a, b = t.split(0.5, seed=3)
    a2, _ = t.split(0.5, seed=3)
    assert [e.state for e in a.examples] == [e.state for e in a2.examples]
    assert not ({e.state["t"] for e in a.examples} & {e.state["t"] for e in b.examples})


def test_mutation_does_not_touch_its_input():
    s = Schema(questions={"q": {"type": "noul", "instructions": "x"}},
               thresholds={"q": 0.5})
    out = mutate_threshold()(s, random.Random(0))
    assert s.thresholds["q"] == 0.5 and out.thresholds["q"] != 0.5


def test_a_perfect_but_tiny_result_is_underpowered_not_refuted():
    """5-0 in favour of the winner cannot reach p<0.05, and must not be
    reported as though the schema had failed."""
    r = Result(best=None, baseline_train=0.0, reported=0.20,
               heldout_baseline=0.0, heldout_best=0.25,
               null_gain=0.012, shrunk=0.12, wins=5, losses=0, p_value=0.0625)
    assert not r.credible
    assert r.underpowered
    assert "UNDERPOWERED" in r.report()
    assert "too small" in r.report()


def test_a_genuinely_flat_result_is_not_called_underpowered():
    r = Result(best=None, baseline_train=0.4, reported=0.4,
               heldout_baseline=0.4, heldout_best=0.4,
               null_gain=0.01, shrunk=0.4, wins=0, losses=0, p_value=1.0)
    assert not r.credible and not r.underpowered
    assert "NOT CREDIBLE" in r.report()


def test_enough_one_sided_pairs_does_reach_significance():
    r = Result(best=None, baseline_train=0.4, reported=0.5,
               heldout_baseline=0.4, heldout_best=0.5,
               null_gain=0.01, shrunk=0.45, wins=6, losses=0, p_value=0.03)
    assert r.credible and not r.underpowered


# --- the public surface: arithmetic on numbers, no model, no dependencies ---

def test_check_catches_a_search_that_found_only_noise():
    """30 candidates that are all equally good still produce a 'winner'."""
    from overtuned import check
    rng = random.Random(0)
    scores = [0.62] + [0.62 + rng.gauss(0, 0.034) for _ in range(29)]
    c = check(scores, n_examples=200)
    assert c.apparent_gain > 0.05          # the search "improved" things
    assert c.floor > 0.05                  # ...by no more than luck would
    assert abs(c.shrunk - 0.62) < 0.03     # de-biasing lands near the truth


def test_check_lets_a_real_winner_through():
    from overtuned import check
    rng = random.Random(1)
    scores = [0.50 + rng.gauss(0, 0.01) for _ in range(20)] + [0.80]
    c = check(scores, n_examples=500)
    assert c.beats_floor and c.shrunk > 0.7


def test_floor_falls_when_each_candidate_is_measured_on_more_data():
    from overtuned import selection_floor
    assert (selection_floor(30, 2000, 0.5, reps=800, seed=1)
            < selection_floor(30, 50, 0.5, reps=800, seed=1) / 2)


def test_confirm_needs_the_pairs_to_line_up():
    from overtuned import confirm
    try:
        confirm([1, 0, 1], [1, 0])
    except ValueError as e:
        assert "paired" in str(e)
    else:
        raise AssertionError("mismatched lengths must not be silently zipped")


def test_confirm_reads_a_clear_win():
    from overtuned import confirm
    c = confirm([0] * 10 + [1] * 10, [1] * 10 + [1] * 10)
    assert c.confirmed and c.wins == 10 and c.losses == 0


def test_importing_the_package_needs_no_heavy_dependency():
    """check() and confirm() are the front door; they must not drag in torch."""
    import subprocess
    import sys
    code = ("import sys, overtuned; overtuned.check([0.5, 0.6], 100); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       cwd=str(Path(__file__).resolve().parents[1]))
    assert r.returncode == 0, r.stderr.decode()[-400:]
