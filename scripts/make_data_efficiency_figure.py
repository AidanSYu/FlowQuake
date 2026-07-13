"""Figure: data-efficiency curve. Test temporal LL vs amount of training history,
native (from scratch) vs few-shot transfer from the leave-one-region-out pooled
model. Shows native collapsing as history shrinks while transfer stays skillful.

Run: python scripts/make_data_efficiency_figure.py --region chile
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="chile")
    args = ap.parse_args()
    res = json.load(open(f"runs/data_efficiency_{args.region}.json"))
    res = [r for r in res if r["native_tll"] is not None]
    res.sort(key=lambda r: r["n_train"])
    n = np.array([r["n_train"] for r in res])

    def arr(key):
        return np.array([r.get(key) if r.get(key) is not None else np.nan for r in res], float)
    nat, zs, few = arr("native_tll"), arr("zeroshot_tll"), arr("few_tll")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(n, nat, "o-", color="#c0392b", label="native (train from scratch)", lw=2, ms=6)
    ax.plot(n, zs, "^--", color="#27ae60", label="zero-shot transfer (no region training)", lw=2, ms=6)
    ax.plot(n, few, "s-", color="#2471a3", label="few-shot transfer (foundation model)", lw=2, ms=6)
    ax.set_xscale("log")
    ax.set_xlabel("training events (M ≥ mc, truncated history)")
    ax.set_ylabel("test temporal log-likelihood  (nats/event)")
    ax.set_title(f"{args.region.capitalize()}: transfer rescues data-poor regimes")
    ax.axhline(0, color="grey", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    out = f"figures/data_efficiency_{args.region}.png"
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"{'n_train':>9}{'native':>10}{'zero':>10}{'few':>10}")
    for r in res:
        def f(v): return f"{v:>10.3f}" if v is not None else f"{'--':>10}"
        print(f"{r['n_train']:>9}{f(r['native_tll'])}{f(r['zeroshot_tll'])}{f(r['few_tll'])}")


if __name__ == "__main__":
    main()
