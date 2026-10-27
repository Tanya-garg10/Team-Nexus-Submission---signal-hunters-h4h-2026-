import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d

ON = (0, 2, 4)
OFF = (1, 3, 5)
# drift grid in channels per row (fine near zero, coarse for steep satellites / radar)
DRIFTS = np.concatenate([np.arange(-3, 3.01, 0.2), [-12, -9, -7, -5.5, -4.5, -3.8, 3.8, 4.5, 5.5, 7, 9, 12]])
DRIFTS = np.unique(np.round(DRIFTS, 3))
ND = len(DRIFTS)


def preprocess(x, mask_center=False):
    """x: (b,6,16,128) uint8 -> z (b,6,16,128) float32 with NaN for missing, ~unit noise."""
    x = x.astype(np.float32)
    x[x == 0] = np.nan
    if mask_center:  # GBT centre spike: replace by neighbours
        x[..., 63:66] = np.nan
    # remove per-row broadband offsets (glitches, gain)
    rowmed = np.nanmedian(x, axis=3, keepdims=True)
    z = x - rowmed
    # per-panel smooth bandpass: median over time, then median filter along frequency
    bp = np.nanmedian(z, axis=2)                       # (b,6,128)
    bp = np.where(np.isnan(bp), 0, bp)
    bps = median_filter(bp, size=(1, 1, 11), mode='nearest')
    z = z - bps[:, :, None, :]
    # robust noise per panel
    med = np.nanmedian(z, axis=(2, 3), keepdims=True)
    mad = np.nanmedian(np.abs(z - med), axis=(2, 3), keepdims=True) * 1.4826 + 1e-3
    z = (z - med) / mad
    if mask_center:
        z[..., 63:66] = 0
    return z, rowmed[..., 0]


def shift_rows(a, s):
    """a (...,128), shift by integer s along last axis (positive = move toward higher channel), NaN fill."""
    out = np.full_like(a, np.nan)
    if s >= 0:
        out[..., s:] = a[..., :128 - s] if s < 128 else out[..., s:]
    else:
        out[..., :128 + s] = a[..., -s:]
    return out


def drift_stat(z):
    """z (b,6,16,128). Returns P (b,6,ND,128): sum along line starting at channel c on row 0 with drift d, / sqrt(n)."""
    b = z.shape[0]
    zz = np.nan_to_num(z, nan=0.0)
    valid = (~np.isnan(z)).astype(np.float32)
    P = np.zeros((b, 6, ND, 128), np.float32)
    for k, d in enumerate(DRIFTS):
        num = np.zeros((b, 6, 128), np.float32)
        cnt = np.zeros((b, 6, 128), np.float32)
        for r in range(16):
            s = int(round(d * r))
            # line at channel c+s on row r  <->  start channel c on row 0
            if abs(s) >= 128:
                continue
            row = zz[:, :, r, :]
            v = valid[:, :, r, :]
            if s >= 0:
                num[..., :128 - s] += row[..., s:]
                cnt[..., :128 - s] += v[..., s:]
            else:
                num[..., -s:] += row[..., :128 + s]
                cnt[..., -s:] += v[..., :128 + s]
        P[:, :, k, :] = num / np.sqrt(np.maximum(cnt, 1))
        P[:, :, k, :][cnt < 6] = 0
    return P


def track_stat(P, gap, sel):
    """Coherent track statistic over panels `sel`. P (b,6,ND,128); gap (b,) slew rows.
    Returns T (b,ND,128) summed (dilated) per-panel stats shifted onto panel-0 start channel, plus the shifted panels."""
    b = P.shape[0]
    Pd = maximum_filter1d(P, size=3, axis=-1, mode='constant', cval=-50)
    out = np.zeros((len(sel), b, ND, 128), np.float32) - 50
    for k, d in enumerate(DRIFTS):
        for j, p in enumerate(sel):
            # start position of panel p row 0 = c0 + d * p*(16+gap)
            sh = np.rint(d * p * (16 + gap)).astype(int)
            for u in np.unique(sh):
                m = sh == u
                a = Pd[m, p, k, :]
                # c0 + u = position in panel p -> value at c0 is a[c0+u]
                if abs(u) >= 128:
                    continue
                tmp = np.full_like(a, -50)
                if u >= 0:
                    tmp[:, :128 - u] = a[:, u:]
                else:
                    tmp[:, -u:] = a[:, :128 + u]
                out[j, m, k, :] = tmp
    return out  # (len(sel), b, ND, 128)


def dedisperse_feats(z, rowmed):
    """Broadband features per panel using a family of curved delays."""
    b = z.shape[0]
    zz = np.nan_to_num(z, nan=0.0)
    f = {}
    # simple broadband stats: row means (excess), channel-to-channel coherence
    rm = np.nanmean(z, axis=3)  # (b,6,16)
    f['rm_max'] = np.nanmax(rm, axis=2)  # (b,6)
    f['rm_std'] = np.nanstd(rm, axis=2)
    # time-series correlation of adjacent coarse frequency bands
    q = zz.reshape(b, 6, 16, 8, 16).mean(4)  # (b,6,16,8)
    q = q - q.mean(2, keepdims=True)
    q = q / (q.std(2, keepdims=True) + 1e-6)
    f['adj_corr'] = (q[..., 1:] * q[..., :-1]).mean(axis=(2, 3))  # (b,6)
    # dedispersion search
    best = np.full((b, 6), -50.0, np.float32)
    best_prof = np.zeros((b, 6), np.float32)
    ch = np.arange(128) / 127.0
    for A in (0, 4, 8, 12, 16, -4, -8, -12, -16):
        for B in (0.3, 1.0, 3.0):
            g = ((1 + B * ch) ** -2 - (1 + B) ** -2) / (1 - (1 + B) ** -2)
            sh = np.rint(A * g).astype(int)  # rows per channel
            ts = np.zeros((b, 6, 16), np.float32)
            cn = np.zeros((b, 6, 16), np.float32)
            for c in range(128):
                s = sh[c]
                col = zz[..., c]
                ts += np.roll(col, -s, axis=2)
            ts /= np.sqrt(128.0)
            m = ts.max(axis=2)
            best = np.maximum(best, m)
    f['dd_best'] = best
    return f


def cadence_features(x, meta_gap, mask_center=False):
    b = x.shape[0]
    z, rowmed = preprocess(x, mask_center)
    P = drift_stat(z)
    F = {}
    gap = meta_gap.astype(int)

    # per-panel best line stats: any drift, small drift, steep drift
    small = np.abs(DRIFTS) <= 1.2
    steep = np.abs(DRIFTS) > 3.5
    zero = np.abs(DRIFTS) < 0.05
    for p in range(6):
        F[f'p{p}_max'] = P[:, p].max(axis=(1, 2))
        F[f'p{p}_small'] = P[:, p][:, small].max(axis=(1, 2))
        F[f'p{p}_steep'] = P[:, p][:, steep].max(axis=(1, 2))
        F[f'p{p}_zero'] = P[:, p][:, zero].max(axis=(1, 2))
        flat = P[:, p].reshape(b, -1)
        a = flat.argmax(1)
        F[f'p{p}_d'] = DRIFTS[a // 128]
        F[f'p{p}_c'] = a % 128
        F[f'p{p}_absd'] = np.abs(F[f'p{p}_d'])
        # number of distinct strong lines (zero-drift columns above 5)
        F[f'p{p}_nlines'] = (P[:, p][:, zero].max(axis=1) > 5).sum(axis=1)
        F[f'p{p}_std'] = np.nanstd(z[:, p], axis=(1, 2))
        F[f'p{p}_kurt'] = np.nanmean(z[:, p] ** 4, axis=(1, 2))

    # cadence track search over ON panels and OFF panels
    TON = track_stat(P, gap, ON)    # (3,b,ND,128)
    TOF = track_stat(P, gap, OFF)
    sON = TON.sum(axis=0)
    sOF = TOF.sum(axis=0)
    # restrict to drifts that keep a physical track in the window
    for name, S, T3 in (('on', sON, TON), ('off', sOF, TOF)):
        flat = S.reshape(b, -1)
        a = flat.argmax(1)
        F[f'{name}_track_max'] = flat.max(1)
        F[f'{name}_track_d'] = DRIFTS[a // 128]
        F[f'{name}_track_c'] = a % 128
        kk, cc = a // 128, a % 128
        for j in range(3):
            F[f'{name}_track_p{j}'] = T3[j][np.arange(b), kk, cc]
        F[f'{name}_track_min'] = np.minimum.reduce([F[f'{name}_track_p{j}'] for j in range(3)])
        # small-drift-only best
        sm = S[:, small].reshape(b, -1)
        F[f'{name}_track_small'] = sm.max(1)
    # OFF response along the best ON track (same start channel, same drift)
    flat = sON.reshape(b, -1)
    a = flat.argmax(1)
    kk, cc = a // 128, a % 128
    for j in range(3):
        F[f'off_on_track_{j}'] = TOF[j][np.arange(b), kk, cc]
    F['off_on_track_max'] = np.maximum.reduce([F[f'off_on_track_{j}'] for j in range(3)])
    # free (non-connecting) per-drift sum vs coherent
    freeON = np.stack([maximum_filter1d(P[:, p], size=1, axis=-1).max(axis=2) for p in ON]).sum(0)  # (b,ND)
    cohON = sON.max(axis=2)
    F['free_on_max'] = freeON.max(1)
    F['gap_free_coh'] = (freeON - cohON).min(1)
    kf = freeON.argmax(1)
    F['free_d'] = DRIFTS[kf]
    F['coh_at_free'] = cohON[np.arange(b), kf]
    F['free_minus_coh'] = F['free_on_max'] - F['coh_at_free']
    # free OFF at same drift
    freeOFF = np.stack([P[:, p].max(axis=2) for p in OFF]).sum(0)
    F['free_off_at_free'] = freeOFF[np.arange(b), kf]

    # broadband / dedispersion
    D = dedisperse_feats(z, rowmed)
    for k, v in D.items():
        for p in range(6):
            F[f'{k}_{p}'] = v[:, p]
    # texture (scintillation): spectral coarse-band variance of time-averaged data
    zz = np.nan_to_num(z, nan=0.0)
    tm = zz.mean(axis=2)  # (b,6,128)
    sm = tm.reshape(b, 6, 16, 8).mean(3)
    for p in range(6):
        F[f'p{p}_patch'] = sm[:, p].std(1)
        F[f'p{p}_tmmax'] = tm[:, p].max(1)
    # broadband level changes between ON and OFF
    lvl = np.nanmean(rowmed, axis=2)  # (b,6)
    for p in range(6):
        F[f'lvl{p}'] = lvl[:, p] - np.nanmean(lvl, axis=1)
    F['gap'] = gap
    names = list(F.keys())
    M = np.stack([np.asarray(F[k], dtype=np.float32) for k in names], axis=1)
    return M, names
