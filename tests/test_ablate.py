"""Tests for the solver-mode ablation.

The script answers one question -- does the clock or the persistent table buy
difficulty 6's strength -- and it answers it by dividing one difference by
another. Arithmetic like that fails quietly: an unbalanced schedule still
produces a percentage, still prints a verdict, and is still wrong.

So what is tested here is the *design*, not the search. ``play_game`` is
stubbed throughout, because a real game at a real clock takes minutes and
proves nothing about the bookkeeping.
"""

from __future__ import annotations

import itertools

import pytest

from scripts import ablate
from scripts.tournament import Outcome


def stub_games(monkeypatch, winner_for):
    """Replace ``play_game`` with something instant and scripted.

    ``winner_for(first, second)`` returns 1, 2 or 0 -- the same convention the
    real ``Outcome`` uses: 1 is the side that moved first.
    """
    calls: list[tuple[ablate.Config, ablate.Config, tuple[int, ...]]] = []

    def fake(first, second, opening, base_time_s):
        calls.append((first, second, opening))
        return Outcome(
            first=0,
            second=0,
            opening=opening,
            winner=winner_for(first, second),
            plies=10,
            seconds=0.0,
        )

    monkeypatch.setattr(ablate, "play_game", fake)
    return calls


def make_report(openings: int = 3) -> ablate.Report:
    return ablate.Report(
        base_time_s=0.01,
        solver_multiple=12.0,
        openings=ablate.make_openings(openings, 4, seed=0),
    )


def test_the_four_configurations_cross_both_knobs():
    """A 2x2 with a corner missing cannot attribute anything to either knob."""
    corners = {(c.multiple > 1, c.persist) for c in ablate.configs(12.0)}
    assert corners == {(False, False), (True, False), (False, True), (True, True)}


def test_every_pairing_is_played_from_both_seats(monkeypatch):
    """Connect 4 is a first-player win, so a one-seat schedule measures seats."""
    calls = stub_games(monkeypatch, lambda first, second: 1)
    report = make_report(openings=2)
    ablate.run(report, verbose=False)

    seatings = {(first.name, second.name) for first, second, _ in calls}
    assert seatings == set(itertools.permutations([c.name for c in ablate.configs(12.0)], 2))


def test_every_configuration_plays_the_same_number_of_games(monkeypatch):
    stub_games(monkeypatch, lambda first, second: 1)
    report = make_report(openings=3)
    ablate.run(report, verbose=False)

    played = {name: record.played for name, record in ablate.records(report).items()}
    assert len(set(played.values())) == 1, played


def test_the_time_bound_stops_on_a_whole_opening_and_stays_balanced(monkeypatch):
    """The point of the bound: a cut run is a smaller experiment, not a skewed one.

    Cutting mid-opening would leave whichever pairings happened to be scheduled
    first with extra games, and the share-of-the-gap arithmetic would divide by
    that without complaining.
    """
    stub_games(monkeypatch, lambda first, second: 1)
    report = make_report(openings=4)
    # Any budget at all is already exceeded, since the stub takes no time but
    # the check runs after the first opening regardless.
    ablate.run(report, verbose=False, max_seconds=0.0)

    pairs = len(list(itertools.permutations(ablate.configs(12.0), 2)))
    assert len(report.outcomes) == pairs
    assert len(report.openings) == 1

    played = {name: record.played for name, record in ablate.records(report).items()}
    assert len(set(played.values())) == 1, played


def test_no_bound_plays_every_opening(monkeypatch):
    stub_games(monkeypatch, lambda first, second: 1)
    report = make_report(openings=3)
    ablate.run(report, verbose=False, max_seconds=None)

    pairs = len(list(itertools.permutations(ablate.configs(12.0), 2)))
    assert len(report.outcomes) == pairs * 3


def test_the_bound_does_not_truncate_the_final_opening(monkeypatch):
    """An exhausted budget on the last opening is not an early stop."""
    stub_games(monkeypatch, lambda first, second: 1)
    report = make_report(openings=1)
    ablate.run(report, verbose=False, max_seconds=0.0)

    assert len(report.openings) == 1
    assert "L5" in ablate.records(report)


def test_a_draw_credits_both_sides_a_half(monkeypatch):
    stub_games(monkeypatch, lambda first, second: 0)
    report = make_report(openings=1)
    ablate.run(report, verbose=False)

    for name, record in ablate.records(report).items():
        assert record.wins == 0 and record.losses == 0, name
        assert record.points == record.played / 2


def test_the_report_refuses_to_divide_by_a_zero_span(monkeypatch):
    """All draws means L5 and L6 did not separate. A share is undefined then."""
    stub_games(monkeypatch, lambda first, second: 0)
    report = make_report(openings=1)
    ablate.run(report, verbose=False)

    text = ablate.format_report(report)
    assert "did not separate" in text
    assert "%" not in text


def test_a_clean_sweep_for_the_clock_reads_as_the_clock(monkeypatch):
    """Rig the results so only the clock matters, and check the verdict follows."""

    def winner(first, second):
        if (first.multiple > 1) == (second.multiple > 1):
            return 0
        return 1 if first.multiple > 1 else 2

    stub_games(monkeypatch, winner)
    report = make_report(openings=2)
    ablate.run(report, verbose=False)

    text = ablate.format_report(report)
    assert "The clock is the bigger of the two here." in text


def test_a_scaled_down_run_says_so_and_a_full_clock_does_not():
    """The caveat is the honest part of the output; losing it would be a bug."""
    scaled = ablate.verdict(0.9, 0.1, games=12, base_time_s=0.1)
    assert any("--time 1.0" in line for line in scaled)

    shipped = ablate.verdict(0.9, 0.1, games=12, base_time_s=1.0)
    assert not any("--time 1.0" in line for line in shipped)


def test_the_standard_error_grows_with_the_square_root_of_the_sample():
    assert ablate.standard_error(4) == pytest.approx(1.0)
    assert ablate.standard_error(400) == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Telling a slow game from a sleeping laptop.


def make_outcome(seconds: float, cpu_seconds: float) -> Outcome:
    return Outcome(
        first=0, second=0, opening=(3,), winner=1, plies=38,
        seconds=seconds, cpu_seconds=cpu_seconds,
    )


def test_a_game_that_spent_its_wall_clock_computing_is_not_flagged():
    assert not ablate.was_suspended(make_outcome(seconds=110.0, cpu_seconds=108.0))


def test_the_real_overnight_anomaly_is_flagged():
    """24,032s of wall clock for a 38-ply game, on a laptop that slept."""
    assert ablate.was_suspended(make_outcome(seconds=24032.2, cpu_seconds=470.0))


def test_a_game_with_no_cpu_reading_is_not_flagged():
    """`cpu_seconds` defaults to zero, and a default is not evidence."""
    assert not ablate.was_suspended(make_outcome(seconds=24032.2, cpu_seconds=0.0))


def test_the_report_says_so_before_it_shows_the_scores(monkeypatch):
    """Whoever reads the timings needs to know the clock lied first."""
    def fake(first, second, opening, base_time_s):
        return make_outcome(seconds=24032.2, cpu_seconds=470.0)

    monkeypatch.setattr(ablate, "play_game", fake)
    report = make_report(openings=1)
    ablate.run(report, verbose=False)

    text = ablate.format_report(report)
    assert "12 of 12 games ran while the machine was suspended" in text
    assert text.index("suspended") < text.index("did not separate")
