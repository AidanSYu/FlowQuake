"""Child-process bookkeeping that survives the parent being killed.

Written after a concrete incident. The scaling sweep spawns `flowquake.train`
workers with plain `subprocess.Popen`; when the sweep was stopped mid-run, the
parent exited and three trainers kept going, holding 11.7 GB between them. Free
memory read 1.0 GB and swap sat at 8.4 of 9.2 GB while nothing that looked like
a job was running -- `pkill -f scaling_curve` had reported success, because the
pattern matches the parent's command line and not the children's.

That is worse than a leak. A run that has been "stopped" still owns the machine,
and the usual diagnostic (is my job running?) says no. This module makes a
stopped run actually stop:

  * every spawned child is registered,
  * SIGTERM / SIGINT / SIGHUP terminate the whole set before the parent exits,
  * `atexit` covers ordinary and exception paths,
  * anything still alive after a grace period gets SIGKILL.

Children are started in their own process group so a signal can be delivered to
the group -- a trainer that has itself spawned dataloader workers takes those
down with it rather than leaving a second generation behind.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import time

__all__ = ["spawn", "reap_children", "install_reaper", "live_children"]

_CHILDREN: list[subprocess.Popen] = []
_INSTALLED = False
GRACE_SECONDS = 5.0


def live_children() -> list[subprocess.Popen]:
    return [p for p in _CHILDREN if p.poll() is None]


def spawn(cmd, env=None, stdout=None, stderr=subprocess.STDOUT,
          **kw) -> subprocess.Popen:
    """`subprocess.Popen` that will be cleaned up if this process dies."""
    install_reaper()
    p = subprocess.Popen(cmd, env=env, stdout=stdout, stderr=stderr,
                         start_new_session=True, **kw)
    _CHILDREN.append(p)
    return p


def _signal_group(p: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, falling back to the child."""
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            p.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def reap_children(*_args) -> None:
    """Terminate every registered child, then SIGKILL whatever is left."""
    alive = live_children()
    if not alive:
        return
    for p in alive:
        _signal_group(p, signal.SIGTERM)
    deadline = time.time() + GRACE_SECONDS
    while time.time() < deadline and any(p.poll() is None for p in alive):
        time.sleep(0.1)
    for p in alive:
        if p.poll() is None:
            _signal_group(p, signal.SIGKILL)
    _CHILDREN.clear()


def install_reaper() -> None:
    """Idempotently arm the exit and signal handlers."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(reap_children)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def handler(signum, frame, _prev=prev):
            reap_children()
            if callable(_prev) and _prev not in (signal.SIG_IGN, signal.SIG_DFL):
                _prev(signum, frame)
            raise SystemExit(128 + signum)

        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            # not the main thread, or the platform refuses this signal
            pass
