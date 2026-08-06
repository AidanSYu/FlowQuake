# Superseded figures

`g3_panel_white_matched_window_uniform.png`
`g3_panel_white_matched_n_uniform.png`

**Both draw an inverted U that does not exist.** They were produced from the
early-stopping checkpoints, whose mc 2.5 anchor was step 200 -- mid-warmup, and
1.0940 nats below that model's own optimum. The apparent +0.7500 rise from mc
2.5 to 2.0 is an artefact of comparing an untrained model against trained ones.

Retrained with early stopping disabled and every checkpoint retained
(`scripts/checkpoint_surface.py`), the rise is **+0.0126 [-0.2814, +0.3707]** --
indistinguishable from zero. See MOONSHOT.md, "THE INTERIOR OPTIMUM DOES NOT
SURVIVE".

The `n_eff_cells` panel is wrong for the same reason: its "5x sharpening"
(167.8 -> 32.4) was the mc 2.5 model finishing training, not responding to
catalog depth. Corrected, FlowQuake gets slightly BROADER (26.2 -> 35.2) while
ETAS sharpens 2.37x.

**Use `moonshot_answer.png` instead** (`scripts/make_moonshot_answer_figure.py`).

These two are kept rather than deleted because the difference between them and
the corrected figure IS the result: it is what a checkpoint-selection rule
confounded with the axis under study does to a curve.
