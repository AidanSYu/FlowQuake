"""The simulation batch size must be reachable, forwarded, and recorded.

`ntest.simulate_windows` has always taken `max_lanes`, but `scaling_curve.py`
never passed it, so every point scored before 2026-08-05 silently used the
16,384 default -- a value chosen for a 48 GB laptop that had once been driven
into 34 GB of swap. On an RTX 5090 that leaves the device at 56% utilisation
using 2 GB of 32 GB, and 65,536 measures 1.56x faster on the mc 2.5 branch that
carries ~69% of a sweep's cost (results/gpu_max_lanes_tuning.txt).

A knob that exists but is unreachable from the CLI is indistinguishable from no
knob at all, so these tests pin the whole path: the default is unchanged, the
flag exists, the PARALLEL scorer forwards it to its workers (the path that
actually runs sweeps, and the easiest one to leave behind), and the chosen value
is written into the artifact.

That last one matters more than it looks. `max_lanes` does not change the
estimand -- lanes are independent -- but it changes how the shared RNG stream is
consumed, so two runs differing only in batch size are separate draws from the
estimator, not a reproduction of each other. Scoring is unseeded, so without the
value in the artifact there is no way to tell afterwards which was which.
"""
import inspect
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

import scaling_curve as sc  # noqa: E402


def test_default_is_unchanged():
    """16,384 must remain the default.

    Every committed target_process.json was produced at this value. A
    device-dependent default would silently change the sampled results based on
    which machine a run landed on.
    """
    assert sc.DEFAULT_MAX_LANES == 16384


def test_score_point_accepts_and_defaults_max_lanes():
    sig = inspect.signature(sc.score_point)
    assert "max_lanes" in sig.parameters
    assert sig.parameters["max_lanes"].default == sc.DEFAULT_MAX_LANES


def test_cli_exposes_max_lanes():
    """The flag has to be reachable, which is the whole point of the change."""
    out = subprocess.run([sys.executable, sc.__file__, "--help"],
                         capture_output=True, text=True).stdout
    assert "--max-lanes" in out


def test_score_many_forwards_max_lanes_to_workers(tmp_path, monkeypatch):
    """The parallel path is the one that runs real sweeps -- pin it directly.

    `score_many` spawns `--score-one` subprocesses. If it omits --max-lanes the
    workers fall back to the default and the tuning is silently lost, which is
    exactly the failure this change fixes.
    """
    captured = []

    class _FakeProc:
        returncode = 0

        def poll(self):
            return 0

    def fake_spawn(cmd, **kw):
        captured.append(cmd)
        return _FakeProc()

    monkeypatch.setattr(sc, "spawn", fake_spawn)
    monkeypatch.setattr(sc, "wait_for_memory", lambda *a, **k: None)

    ck = tmp_path / "pt" / "ckpt_best.pt"
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b"")

    sc.score_many([ck], tmp_path, n_sims=200, sample_steps=16, device="cuda",
                  concurrency=1, max_lanes=65536)

    assert len(captured) == 1, "expected exactly one worker spawn"
    cmd = captured[0]
    assert "--max-lanes" in cmd, f"--max-lanes missing from worker cmd: {cmd}"
    assert cmd[cmd.index("--max-lanes") + 1] == "65536"


def test_artifact_records_batch_size_and_device():
    """Provenance: without these the artifact cannot be placed.

    Checked on the source of `score_point` rather than by running it, because
    running it needs a trained checkpoint and a catalog. The assertion is still
    meaningful -- it fails if someone drops the keys from the result dict.
    """
    src = inspect.getsource(sc.score_point)
    assert '"max_lanes"' in src
    assert '"device"' in src
