import numpy as np
from feats import preprocess, ON, OFF

DF = np.unique(np.round(np.concatenate([
    np.arange(-0.6, 0.6001, 0.01),
    np.arange(0.64, 3.01, 0.04), -np.arange(0.64, 3.01, 0.04)]), 3))


def track_features(z, gap):
    """z (b,6,16,128) preprocessed (NaN = missing). Coherent line search over global time with fine drift grid."""
    b = z.shape[0]
    # light smoothing along frequency (lines ~1-2 channels wide)
    zz = np.nan_to_num(z, nan=0.0)
    vm = (~np.isnan(z)).astype(np.float32)
    zs = zz.copy()
    zs[..., 1:-1] = (zz[..., :-2] + 2 * zz[..., 1:-1] + zz[..., 2:]) / 4.0 * 1.0
    vs = vm.copy()
    ar = np.arange(16)[:, None]
    c0 = np.arange(128)[None, :]
    out = {}
    names = []
    res = {k: np.zeros(b, np.float32) for k in
           ['on_best', 'on_best_d', 'on_best_c', 'off_best', 'off_best_d', 'min_best', 'min_best_d', 'zero_on', 'zero_min',
            'zero_off', 'free_on', 'free_on_d', 'free_min_nodrift_on', 'on_best_nz']}
    for j in range(3):
        res[f'on_at_{j}'] = np.zeros(b, np.float32)
        res[f'off_at_{j}'] = np.zeros(b, np.float32)
        res[f'minb_off_{j}'] = np.zeros(b, np.float32)
        res[f'minb_on_{j}'] = np.zeros(b, np.float32)
    res['on_best'][:] = -50; res['off_best'][:] = -50; res['min_best'][:] = -50; res['free_on'][:] = -50
    res['on_best_nz'][:] = -50
    # per-drift free statistic needs the best coherent value at the same drift
    for g in np.unique(gap):
        sel = np.where(gap == g)[0]
        zg, vg, sg = zs[sel], vm[sel], None
        nb = len(sel)
        for d in DF:
            Sp = np.zeros((6, nb, 128), np.float32)
            for p in range(6):
                t = p * (16 + g) + np.arange(16)
                sh = np.rint(d * t).astype(int)[:, None]
                idx = c0 + sh
                ok = (idx >= 0) & (idx < 128)
                idc = np.clip(idx, 0, 127)
                vals = zg[:, p][:, ar, idc] * ok
                cnt = (vg[:, p][:, ar, idc] * ok).sum(1)
                s = vals.sum(1) / np.sqrt(np.maximum(cnt, 1.0))
                s[cnt < 8] = 0.0
                Sp[p] = s
            on = Sp[[0, 2, 4]]
            of = Sp[[1, 3, 5]]
            son = on.sum(0) / np.sqrt(3.0)
            sof = of.sum(0) / np.sqrt(3.0)
            smin = on.min(0)
            # best ON track at this drift
            a = son.argmax(1); r = np.arange(nb); v = son[r, a]
            u = v > res['on_best'][sel]
            if u.any():
                ii = sel[u]; aa = a[u]; rr = r[u]
                res['on_best'][ii] = v[u]; res['on_best_d'][ii] = d; res['on_best_c'][ii] = aa
                for j in range(3):
                    res[f'on_at_{j}'][ii] = on[j][rr, aa]
                    res[f'off_at_{j}'][ii] = of[j][rr, aa]
            if abs(d) > 0.045:
                u2 = v > res['on_best_nz'][sel]
                res['on_best_nz'][sel[u2]] = v[u2]
            if abs(d) < 1e-6:
                res['zero_on'][sel] = v
                res['zero_min'][sel] = smin.max(1)
                res['zero_off'][sel] = sof.max(1)
            a = sof.argmax(1); v = sof[r, a]
            u = v > res['off_best'][sel]
            res['off_best'][sel[u]] = v[u]; res['off_best_d'][sel[u]] = d
            a = smin.argmax(1); v = smin[r, a]
            u = v > res['min_best'][sel]
            if u.any():
                ii = sel[u]; aa = a[u]; rr = r[u]
                res['min_best'][ii] = v[u]; res['min_best_d'][ii] = d
                for j in range(3):
                    res[f'minb_off_{j}'][ii] = of[j][rr, aa]
                    res[f'minb_on_{j}'][ii] = on[j][rr, aa]
            # free: each ON panel picks its own best position (same drift)
            fr = (on.max(2)).sum(0) / np.sqrt(3.0)
            u = fr > res['free_on'][sel]
            res['free_on'][sel[u]] = fr[u]; res['free_on_d'][sel[u]] = d
    f = {}
    f.update(res)
    f['on_minus_zero'] = res['on_best'] - res['zero_on']
    f['min_minus_zero'] = res['min_best'] - res['zero_min']
    f['coh_gap'] = res['free_on'] - res['on_best']
    f['off_max_at_on'] = np.maximum.reduce([res[f'off_at_{j}'] for j in range(3)])
    f['off_max_at_min'] = np.maximum.reduce([res[f'minb_off_{j}'] for j in range(3)])
    f['on_min_at'] = np.minimum.reduce([res[f'on_at_{j}'] for j in range(3)])
    f['abs_d'] = np.abs(res['on_best_d'])
    f['abs_dmin'] = np.abs(res['min_best_d'])
    names = list(f.keys())
    M = np.stack([np.asarray(f[k], np.float32) for k in names], 1)
    return M, names
