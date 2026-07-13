"""
Localize the spatial LL deficit: FlowQuake sll vs ETAS SLL on ComCat_25 test.
Define per-event spatial gain g = our_sll - etas_SLL  (negative => we lose).
"""
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
np.set_printoptions(suppress=True)

OURS = 'runs/comcat25_s1555/per_event_test.csv'
ETAS = 'reference/Experiments/ETAS/output_data_ComCat_25/augmented_catalog.csv'

ours = pd.read_csv(OURS, parse_dates=['time'])
etas = pd.read_csv(ETAS, parse_dates=['time'])

# ETAS augmented catalog spans full history; keep only test-window rows (SLL finite).
etas_test = etas[np.isfinite(etas.SLL)].copy()

# Merge on exact timestamp.
df = ours.merge(
    etas_test[['time', 'magnitude', 'x', 'y', 'longitude', 'latitude', 'TLL', 'SLL']],
    on='time', how='inner', validate='one_to_one'
)
assert len(df) == len(ours), f"merge dropped rows: {len(df)} vs {len(ours)}"

df = df.sort_values('time').reset_index(drop=True)

# Per-event spatial gain (nats). Negative = FlowQuake worse than ETAS.
df['g'] = df['sll'] - df['SLL']

N = len(df)
total_gain = df['g'].sum()
mean_gain = df['g'].mean()
print('=' * 90)
print(f'N events = {N}')
print(f'FlowQuake mean sll = {df.sll.mean():.4f}   ETAS mean SLL = {df.SLL.mean():.4f}')
print(f'mean spatial gain g (our-ETAS) = {mean_gain:.4f} nats/event   (matches reported -0.40 deficit)')
print(f'total spatial gain = {total_gain:.1f} nats   (sum over all events)')
print(f'frac events where we LOSE (g<0) = {(df.g<0).mean():.3f}   WIN (g>0) = {(df.g>0).mean():.3f}')
print('=' * 90)


def strat(label, groupcol, df):
    print(f'\n--- {label} ---')
    grp = df.groupby(groupcol, observed=True)
    out = grp['g'].agg(['count', 'mean', 'sum']).rename(
        columns={'count': 'N', 'mean': 'mean_g', 'sum': 'total_g'})
    out['frac_of_deficit'] = out['total_g'] / total_gain  # share of total (signed) gain
    out['pct_events'] = out['N'] / N * 100
    out['our_sll'] = grp['sll'].mean()
    out['etas_sll'] = grp['SLL'].mean()
    print(out.to_string(float_format=lambda v: f'{v:9.3f}'))
    return out


# ---------------------------------------------------------------------------
# 1) Magnitude bins
# ---------------------------------------------------------------------------
mbins = [2.5, 3.0, 4.0, 5.0, 10.0]
mlab = ['2.5-3', '3-4', '4-5', '5+']
df['mbin'] = pd.cut(df.magnitude, bins=mbins, labels=mlab, right=False, include_lowest=True)
strat('1. MAGNITUDE bins', 'mbin', df)

# ---------------------------------------------------------------------------
# 2) Preceding inter-event time tau (days), from sorted catalog time diffs
# ---------------------------------------------------------------------------
dt_days = df['time'].diff().dt.total_seconds() / 86400.0
df['tau'] = dt_days
df['tau_filled'] = df['tau'].fillna(df['tau'].median())  # first event
taubins = [0, 1e-3, 1e-2, 0.1, 1.0, 1e9]
taulab = ['<0.001d (~min)', '0.001-0.01d', '0.01-0.1d', '0.1-1d', '>1d (bkg)']
df['taubin'] = pd.cut(df.tau_filled, bins=taubins, labels=taulab, right=False)
strat('2. PRECEDING inter-event time tau (aftershock<0.1d vs background>1d)', 'taubin', df)

# coarse aftershock vs background
df['regime'] = np.where(df.tau_filled < 0.1, 'aftershock(<0.1d)',
                np.where(df.tau_filled > 1.0, 'background(>1d)', 'mid(0.1-1d)'))
strat('2b. coarse regime', 'regime', df)

# ---------------------------------------------------------------------------
# 3) The 50 single worst-loss events
# ---------------------------------------------------------------------------
print('\n--- 3. 50 WORST single-event spatial losses (most negative g) ---')
worst = df.nsmallest(50, 'g').copy()
print(f'these 50 events ({50/N*100:.2f}% of catalog) sum g = {worst.g.sum():.1f} nats '
      f'= {worst.g.sum()/total_gain*100:.1f}% of total deficit')
print(f'their mean magnitude = {worst.magnitude.mean():.2f} (catalog mean {df.magnitude.mean():.2f})')
print(f'their median tau = {worst.tau_filled.median():.4f} d (catalog median {df.tau_filled.median():.4f} d)')
print(f'frac of worst-50 that are aftershock-like (tau<0.1d) = {(worst.tau_filled<0.1).mean():.2f}')
print(f'frac of worst-50 that are background-like (tau>1d)  = {(worst.tau_filled>1.0).mean():.2f}')
cols = ['time', 'magnitude', 'x', 'y', 'tau_filled', 'sll', 'SLL', 'g']
with pd.option_context('display.max_rows', 60):
    print(worst[cols].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

# ---------------------------------------------------------------------------
# 4) Spatial: local catalog density + geographic bins
# ---------------------------------------------------------------------------
print('\n--- 4. SPATIAL: local catalog density (dense fault vs diffuse/offshore) ---')
# Local density = # of catalog events within R km of each event (using x,y in km).
xy = df[['x', 'y']].to_numpy()
R = 10.0  # km
# Block computation to keep memory bounded.
dens = np.zeros(N, dtype=np.int32)
B = 2000
x = xy[:, 0]; y = xy[:, 1]
for i0 in range(0, N, B):
    i1 = min(i0 + B, N)
    dx = x[i0:i1, None] - x[None, :]
    dy = y[i0:i1, None] - y[None, :]
    d2 = dx * dx + dy * dy
    dens[i0:i1] = (d2 <= R * R).sum(axis=1) - 1  # exclude self
df['local_density'] = dens
qlab = ['Q1 sparse', 'Q2', 'Q3', 'Q4 dense']
df['densbin'] = pd.qcut(df.local_density, q=4, labels=qlab, duplicates='drop')
strat(f'4a. local density quartile (# events within {R}km)', 'densbin', df)

# distance from a recent-activity proxy is captured by tau; here add nearest-neighbor distance in space
print('\n--- 4b. nearest spatial neighbor distance (isolation in space) ---')
nn = np.full(N, np.inf)
for i0 in range(0, N, B):
    i1 = min(i0 + B, N)
    dx = x[i0:i1, None] - x[None, :]
    dy = y[i0:i1, None] - y[None, :]
    d2 = dx * dx + dy * dy
    np.fill_diagonal_indices = None
    # set self to inf
    for k in range(i0, i1):
        d2[k - i0, k] = np.inf
    nn[i0:i1] = np.sqrt(d2.min(axis=1))
df['nn_km'] = nn
nnbins = [0, 1, 5, 20, 1e9]
nnlab = ['<1km', '1-5km', '5-20km', '>20km (remote)']
df['nnbin'] = pd.cut(df.nn_km, bins=nnbins, labels=nnlab, right=False)
strat('4b. nearest-neighbor distance bin', 'nnbin', df)

# geographic coarse grid (50 km cells), report worst cells
print('\n--- 4c. worst geographic 50km cells (total deficit) ---')
df['gx'] = (df.x // 50).astype(int)
df['gy'] = (df.y // 50).astype(int)
cell = df.groupby(['gx', 'gy']).agg(N=('g', 'size'), total_g=('g', 'sum'),
                                    mean_g=('g', 'mean'),
                                    cx=('x', 'mean'), cy=('y', 'mean')).reset_index()
cell = cell[cell.N >= 20]
worst_cells = cell.nsmallest(12, 'total_g')
print(worst_cells[['cx', 'cy', 'N', 'mean_g', 'total_g']].to_string(
    index=False, float_format=lambda v: f'{v:9.2f}'))
print(f'top-12 worst cells account for {worst_cells.total_g.sum()/total_gain*100:.1f}% of total deficit '
      f'with {worst_cells.N.sum()/N*100:.1f}% of events')

# ---------------------------------------------------------------------------
# 5) Concentration of the deficit
# ---------------------------------------------------------------------------
print('\n--- 5. CONCENTRATION of the deficit (is it a few % of events?) ---')
g_sorted = np.sort(df.g.values)  # most negative first
cum = np.cumsum(g_sorted)
for frac in [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
    k = int(frac * N)
    share = cum[k - 1] / total_gain * 100
    print(f'worst {frac*100:5.1f}% of events ({k:5d}) account for {share:6.1f}% of total deficit '
          f'(their mean g = {g_sorted[:k].mean():.3f})')
# how many events would we need to "fix" (set g=0) to close the gap?
neg = df.g[df.g < 0].sort_values()
print(f'\nsum of all NEGATIVE g (pure losses) = {neg.sum():.1f} nats over {len(neg)} events')
print(f'sum of all POSITIVE g (pure wins)  = {df.g[df.g>0].sum():.1f} nats over {(df.g>0).sum()} events')

# cross-tab: where does deficit live -- magnitude x regime
print('\n--- bonus: magnitude x regime total_g cross-tab (nats) ---')
ct = df.pivot_table(index='mbin', columns='regime', values='g', aggfunc='sum', observed=True)
print(ct.to_string(float_format=lambda v: f'{v:9.2f}'))
print('\n--- bonus: magnitude x regime mean_g cross-tab (nats/event) ---')
ctm = df.pivot_table(index='mbin', columns='regime', values='g', aggfunc='mean', observed=True)
print(ctm.to_string(float_format=lambda v: f'{v:9.3f}'))
print('\n--- bonus: magnitude x regime count cross-tab ---')
ctn = df.pivot_table(index='mbin', columns='regime', values='g', aggfunc='size', observed=True)
print(ctn.to_string())

# Decompose total deficit into additive contributions for the ranked summary.
print('\n=== DEFICIT DECOMPOSITION (total nats by mutually-exclusive populations) ===')
df['popbucket'] = 'other'
# Assign each event to ONE bucket by priority to avoid double counting.
df.loc[df.magnitude >= 5.0, 'popbucket'] = 'mainshock M5+'
df.loc[(df['popbucket'] == 'other') & (df.magnitude >= 4.0), 'popbucket'] = 'M4-5'
df.loc[(df['popbucket'] == 'other') & (df.tau_filled < 0.1), 'popbucket'] = 'aftershock M<4 (tau<0.1d)'
df.loc[(df['popbucket'] == 'other') & (df.tau_filled > 1.0), 'popbucket'] = 'background M<4 (tau>1d)'
df.loc[df['popbucket'] == 'other', 'popbucket'] = 'mid M<4 (0.1-1d)'
decomp = df.groupby('popbucket').agg(N=('g', 'size'), total_g=('g', 'sum'), mean_g=('g', 'mean'))
decomp['share_%'] = decomp.total_g / total_gain * 100
decomp = decomp.sort_values('total_g')
print(decomp.to_string(float_format=lambda v: f'{v:9.3f}'))
print(f'\nTOTAL = {total_gain:.1f} nats over {N} events ({mean_gain:.4f}/event)')
