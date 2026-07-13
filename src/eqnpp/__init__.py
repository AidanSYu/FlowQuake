"""eqnpp — selective-SSM whole-catalog encoder + flow-matching marked TPP for EarthquakeNPP.

Goal: beat the 40-year ETAS incumbent on the EarthquakeNPP benchmark, where the decisive
metric is the per-event spatial log-likelihood (SLL). On ComCat_25, ETAS scores SLL = -8.69
and every published neural point process loses on spatial. The two reasons NPPs lose map onto
the two levers this package pulls:

  1. fixed-window encoders  ->  selective-SSM (Mamba/S6) whole-catalog encoder (unbounded history)
  2. hand-crafted Omori/GR kernels  ->  learned flow density over (Δt, x, y, m)

See README.md for the experiment design and the ablation ladder that isolates each lever.
"""

__all__ = ["data", "ssm", "flows", "heads", "model", "metrics"]
