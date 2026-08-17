"""Fitting magnitude decoupled from the ladder's top rung.

WHY THIS KNOB EXISTS. The ladder's headline is a SHAPE: marginal forecast value
shrinks as events get smaller. A referee's cheapest attack is that the shape
belongs to ETAS rather than to the catalogue -- that one particular frozen theta
produces it and another would not. Answering that means holding the rungs,
targets and windows fixed while varying only which events the fit saw.

The original script could not do it. `--anchor` set both the fit threshold and
the ladder's top rung, so changing the fit necessarily changed the measurement
range, and a genuinely stable result was indistinguishable from a coincidence.

THE INVARIANT THAT MATTERS MOST. Omitting `--fit-anchor` must reproduce the old
behaviour exactly, because every published number was produced under it. A
control that silently shifts the baseline it is meant to defend is worse than no
control.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _parse(argv):
    """Reach the argument parser without running an hour of ETAS."""
    import argparse

    import scripts.information_ladder as il
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        ns = real(self, args, namespace)
        captured["ns"] = ns
        raise SystemExit(0)          # stop before any work happens

    argparse.ArgumentParser.parse_args = spy
    try:
        with pytest.raises(SystemExit):
            il.main(argv)
    finally:
        argparse.ArgumentParser.parse_args = real
    return captured["ns"]


def test_fit_anchor_defaults_to_none_so_published_runs_are_unchanged():
    ns = _parse(["--anchor", "2.5", "--floor", "1.0"])
    assert ns.fit_anchor is None
    # The resolution rule the script applies: None means "use --anchor".
    assert (ns.fit_anchor if ns.fit_anchor is not None else ns.anchor) == 2.5


def test_fit_anchor_overrides_only_the_fit():
    ns = _parse(["--anchor", "2.5", "--floor", "1.0", "--fit-anchor", "2.75"])
    assert ns.fit_anchor == 2.75
    assert ns.anchor == 2.5          # the ladder top is untouched
    assert ns.floor == 1.0


def test_the_ladder_rungs_depend_on_anchor_not_fit_anchor():
    """The whole point: the measurement range must not move with the fit."""
    def rungs(ns):
        n = int(round((ns.anchor - ns.floor) / ns.step))
        return [round(ns.anchor - i * ns.step, 4) for i in range(n + 1)]

    base = rungs(_parse(["--anchor", "2.5", "--floor", "1.0", "--step", "0.25"]))
    for fa in ("2.25", "2.75", "3.0"):
        got = rungs(_parse(["--anchor", "2.5", "--floor", "1.0", "--step", "0.25",
                            "--fit-anchor", fa]))
        assert got == base, f"fit-anchor {fa} moved the rungs to {got}"
    assert base == [2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0]


def test_output_records_which_fit_produced_it():
    """A specificity sweep is unreadable if the runs cannot be told apart."""
    src = (ROOT / "scripts" / "information_ladder.py").read_text()
    assert '"fit_anchor": fit_mc' in src


def test_the_fit_selection_uses_fit_mc_not_anchor():
    """Guards the actual bug this refactor could introduce.

    If the event selection kept using args.anchor while the printout said
    fit_mc, every specificity run would silently fit the same events and the
    control would return a reassuring null for the wrong reason.
    """
    src = (ROOT / "scripts" / "information_ladder.py").read_text()
    assert "sel = (mm >= fit_mc)" in src
    assert "mc=fit_mc" in src
    assert "sel = (mm >= args.anchor)" not in src
