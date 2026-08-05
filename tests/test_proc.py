"""Tests for child-process cleanup.

The incident: the scaling sweep was stopped, `pkill -f scaling_curve` reported
success, and three orphaned `flowquake.train` children kept running with 11.7 GB
between them -- free memory 1.0 GB, swap 8.4 of 9.2 GB -- while no job appeared
to be running. `pkill -f` matches the PARENT's command line, so the children
survive it. These tests lock in that a stopped run actually stops.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from flowquake import proc


@pytest.fixture(autouse=True)
def _clean_registry():
    proc._CHILDREN.clear()
    yield
    proc.reap_children()
    proc._CHILDREN.clear()


def _sleeper(seconds=60):
    return proc.spawn([sys.executable, "-c", f"import time; time.sleep({seconds})"],
                      stdout=subprocess.DEVNULL)


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_spawned_child_is_registered_and_reaped():
    p = _sleeper()
    assert p in proc.live_children()
    proc.reap_children()
    assert p.poll() is not None
    assert proc.live_children() == []


def test_reap_kills_every_child_not_just_the_first():
    """The incident left THREE trainers behind, so one-child cleanup is not enough."""
    ps = [_sleeper() for _ in range(3)]
    assert len(proc.live_children()) == 3
    proc.reap_children()
    for p in ps:
        assert p.poll() is not None, "a child survived reaping"


def test_child_runs_in_its_own_process_group():
    """So a signal reaches the child's OWN children too, not just the child.

    A trainer that has spawned dataloader workers must take them down with it,
    or reaping the first generation just leaves a second one holding the RAM.
    """
    p = _sleeper()
    assert os.getpgid(p.pid) != os.getpgid(os.getpid())


def test_reap_is_safe_when_nothing_is_running():
    proc.reap_children()
    proc.reap_children()


def test_reap_tolerates_an_already_dead_child():
    p = proc.spawn([sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL)
    p.wait(timeout=30)
    proc.reap_children()


def test_parent_death_takes_the_children_with_it():
    """End-to-end: the exact shape of the incident.

    A parent spawns a long-lived child through this module and is then killed
    the way the sweep was. The child must not outlive it.
    """
    script = (
        "import subprocess, sys, time\n"
        "from flowquake.proc import spawn\n"
        "p = spawn([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "          stdout=subprocess.DEVNULL)\n"
        "print(p.pid, flush=True)\n"
        "time.sleep(120)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        child_pid = int(parent.stdout.readline().strip())
        assert _alive(child_pid)
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=30)
        deadline = time.time() + 15
        while time.time() < deadline and _alive(child_pid):
            time.sleep(0.2)
        assert not _alive(child_pid), (
            f"child {child_pid} outlived its parent -- this is the orphan "
            "incident reproducing")
    finally:
        if parent.poll() is None:
            parent.kill()
