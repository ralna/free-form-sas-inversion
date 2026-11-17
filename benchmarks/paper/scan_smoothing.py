"""
Scan the noise- and discretization-scaled smoothing parameter vs, for which the
smoothing residuals are vs * sqrt(m/(n-1)) * n * diff(w). With one block (or
blocks of nearly equal size) this equals the existing sigma = vs*n*sqrt(m/(n-1)).
Saves w, chi^2 and times to $FFSI_PAPER_DIR/results/smoothing_scan/.
Run after bench_real.py, bench_sphere.py and bench_1d_aniso.py (which save the problems).
"""
import json
import time
import numpy as np
import cupy as cp
from ffsi.optimize_galahad import optimize
from common import PAPER_DIR, PROBLEM_DIR, chi2, sync
from ffsi.models.sphere import Sphere


def build(name):
    """Return (G, I, S, q_axes) on the GPU for a saved 1D problem."""
    P = np.load(PROBLEM_DIR / f'{name}.npz')
    I, S = cp.asarray(P['I_data']), cp.asarray(P['I_data_std'])
    if 'G' in P.files:  # orientation-averaged problems store G
        return cp.asarray(P['G']), I, S, [0]
    G = Sphere.compute_scattering_intensity([cp.asarray(P['q'])], [cp.asarray(P['r'])],
                                            float(P['drho']))
    return G, I, S, [0]


OUT = PAPER_DIR / 'results' / 'smoothing_scan'
OUT.mkdir(parents=True, exist_ok=True)
VS = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
summary = {}
for pname in ('sphere_SANS', 'sphere_SAXS', 'sphere_benchmark', 'cylinder1d'):
    G, I, S, qa = build(pname)
    m = I.size
    n = int(np.mean(G.shape[len(qa):]))
    for vs in VS:
        sigma = None if vs == 0 else vs * n * np.sqrt(m / (n - 1))
        sync(cp); t0 = time.perf_counter()
        xi, b, w = optimize(G, I, S, sigma=sigma, user_options={'print_level': 0})
        sync(cp); dt = time.perf_counter() - t0
        c2, I_opt = chi2(G, w, xi, b, I, S, qa)
        R = [float(len(wk)**2 / (len(wk) - 1) * np.sum(np.diff(wk)**2)) for wk in w]
        np.savez(OUT / f'{pname}_vs{vs:g}.npz', I_opt=I_opt, **{f'w{k}': wk for k, wk in enumerate(w)})
        summary.setdefault(pname, {})[f'{vs:g}'] = dict(chi2=c2, chi2_red=c2 / m, time=dt, R=R, sigma=sigma)
        print(f'@@ {pname:18s} vs={vs:<6g} sigma={sigma if sigma else 0:9.3g} chi2/m={c2/m:.4g} '
              f'R={["%.3g" % r for r in R]} time={dt:.2f}s', flush=True)
        json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
