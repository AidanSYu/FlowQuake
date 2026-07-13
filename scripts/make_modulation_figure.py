"""Figure: what the neural-ETAS head learned beyond ETAS.

Panel A/B/C: the per-parent modulation surfaces g(m_j, dt) -- weight, kernel
scale, tail exponent -- over the (parent magnitude, elapsed time) plane.
Panel D: learned background mixture (uniform vs causal KDE bandwidths).

Run: python scripts/make_modulation_figure.py runs/neural_etas/ComCat_25/head_full_s0.pt
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flowquake.neural_etas import NeuralETASSpatialHead

KDE_BWS = [1.5, 6.0, 25.0, 100.0]


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/neural_etas/ComCat_25/head_full_s0.pt"
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = NeuralETASSpatialHead(ck["params"], n_kde=ck["n_kde"], use_mlp=ck["use_mlp"])
    model.load_state_dict(ck["state"])
    model.eval()
    mc = float(model.mc)

    mags = np.linspace(mc, mc + 5.0, 120)
    dts = np.logspace(-3, 4, 160)  # days
    M, T = np.meshgrid(mags, dts, indexing="ij")
    with torch.no_grad():
        feats = torch.stack([(torch.tensor(M, dtype=torch.float32) - mc) / 2.0,
                             (torch.log(torch.tensor(T, dtype=torch.float32) + 1e-3) - 2.0) / 3.0], dim=-1)
        mods = model.mlp(feats).numpy()
    titles = [r"$\Delta \log$ weight  $g_w(m_j,\Delta t)$",
              r"$\Delta \log$ kernel scale $d$",
              r"$\Delta \log$ tail exponent $\rho$"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for k in range(3):
        ax = axes[k]
        lim = np.abs(mods[..., k]).max() or 1.0
        pc = ax.pcolormesh(T, M, mods[..., k], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set_xscale("log")
        ax.set_xlabel(r"elapsed time $\Delta t$ (days)")
        if k == 0:
            ax.set_ylabel(r"parent magnitude $m_j$")
        ax.set_title(titles[k], fontsize=10)
        fig.colorbar(pc, ax=ax)

    ax = axes[3]
    gate = float(torch.sigmoid(model.kde_gate))
    kw = (gate * torch.softmax(model.kde_logits, 0)).detach().numpy()
    labels = ["uniform"] + [f"KDE {bw:g} km" for bw in KDE_BWS[:len(kw)]]
    vals = [1.0 - gate] + list(kw)
    ax.bar(range(len(vals)), vals, color=["#888"] + ["#c0392b", "#e67e22", "#2980b9", "#27ae60"][:len(kw)])
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_title("learned background mixture\n(ETAS: 100% uniform)", fontsize=10)
    ax.set_ylabel("weight")

    fig.suptitle("Neural-ETAS spatial head: learned departures from the inverted ETAS density "
                 f"(alpha_far={float(torch.exp(model.log_alpha)):.2f}, mu x{float(torch.exp(model.log_mu_adj)):.2f})",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = "figures/fig_neural_etas_modulation.png"
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, dpi=180)
    print("wrote", out)
    print(json.dumps({"bg": dict(zip(labels, [round(float(v), 4) for v in vals])),
                      "alpha": float(torch.exp(model.log_alpha)),
                      "mu_adj": float(torch.exp(model.log_mu_adj))}, indent=2))


if __name__ == "__main__":
    main()
