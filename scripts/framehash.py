"""Canonical fingerprint of a scoring frame. Same bytes => same experiment.

Hashes the target observations exactly (float64 buffer, no text formatting) plus
the grid/spec fields that decide what a score MEANS. Two frames agreeing here
are interchangeable; disagreeing anywhere here makes a contrast meaningless.
"""
import hashlib, json, sys

import numpy as np

f = json.load(open(sys.argv[1]))
h = hashlib.sha256()
for w in f["windows"]:
    h.update(np.asarray(w["obs"], dtype=np.float64).tobytes())
    h.update(np.float64(w["start_days"]).tobytes())
h.update(json.dumps(f["grid"], sort_keys=True).encode())
h.update(json.dumps(f["spec"], sort_keys=True).encode())
h.update(np.asarray(f["active_cells"], dtype=np.int64).tobytes())
h.update(str(f.get("mc_ref")).encode())
print(f"obs_hash {h.hexdigest()[:16]}  windows {f['n_windows']} "
      f"targets {f['n_target_events']} cells {f['n_active_cells']} "
      f"mc_ref {f.get('mc_ref')} b {f['b_value']:.6f} bin {f['grid']['bin_km']}")
