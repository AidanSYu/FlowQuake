# 12 · EarthquakeNPP — beat ETAS

**Standalone mechanism (selective-SSM + flow-matching marked point process) · P1.2 · top score**
Compute: **4090** · score **23 (top)**

---

## The bet
Topple a **40-year physical incumbent (ETAS)** on its own CSEP tests. The two reasons neural point-processes lose to ETAS — fixed-window encoders + hand-crafted Omori/Gutenberg-Richter kernels — map exactly onto the two things this stack replaces. **Selective-SSM whole-catalog encoder + flow-matching marked point process.**

## Kill (~2 wk, 4090)
On **ComCat**, swap DeepSTPP's encoder for a Mamba full-catalog encoder + magnitude mark, run **temporal N-test**. **BUT the temporal win is known to underdeliver** (benchmark authors tried it) — you **MUST win on CSEP spatial/magnitude**, not temporal LL.

## Internal chain (one project, escalating)
temporal kill → **spatial/magnitude CSEP win** → inject a **neural-operator Coulomb-stress kernel** as the anisotropic spatial intensity (the one axis ETAS still wins).

## Baselines to beat
ETAS (CSEP spatial/magnitude), DeepSTPP.

## First steps
- [ ] ComCat catalog + EarthquakeNPP harness
- [ ] Mamba whole-catalog encoder + magnitude mark, FM marked TPP
- [ ] Temporal N-test → CSEP spatial/magnitude → Coulomb-stress kernel
