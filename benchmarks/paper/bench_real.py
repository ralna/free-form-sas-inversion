"""
Paper benchmark: real SANS and SAXS data from polydisperse spheres
(parameters identical to ffsi/scripts/sas_inversion_sphere_real_data.py)

Copyright (C) 2026 The Science and Technology Facilities Council (STFC)
Author: Jaroslav Fowkes (STFC)
"""
import numpy as np
import cupy as cp
from ffsi.models.sphere import Sphere
from common import REPO_ROOT, timed, save_problem, run_inversion, save_result

drho = 1

DATASETS = {
    # name: (file, high-q truncation, r lower, r upper, nr, smoothing sigma)
    'SANS': (REPO_ROOT / 'ffsi/data/SANS/observation.txt', 285, 400, 800, 1000, 5.0),
    'SAXS': (REPO_ROOT / 'ffsi/data/SAXS/observation_corrected.txt', None, 400, 1200, 1000, 10.0),
}

for dataset, (fname, tr, rl, ru, nr, SIGMA) in DATASETS.items():
    data = np.loadtxt(fname)[:tr]
    q, I_data, I_data_std = data[:, 0], data[:, 1], data[:, 2]
    r = np.linspace(rl, ru, nr)
    save_problem(f'sphere_{dataset}', q=q, r=r, drho=drho, I_data=I_data, I_data_std=I_data_std)

    for device, xp in (('gpu', cp), ('cpu', np)):
        q_d, r_d, I_d, S_d = (xp.asarray(a) for a in (q, r, I_data, I_data_std))
        G_d, t_G = timed(xp, Sphere.compute_scattering_intensity, [q_d], [r_d], drho)
        for sigma, tag in ((SIGMA, 'reg'), (None, 'noreg')):
            res = run_inversion(G_d, I_d, S_d, sigma, q_axes=[0])
            save_result(f'sphere_{dataset}_{device}_{tag}', res, G_time=t_G,
                        sigma=sigma, device=device, nq=len(q))
