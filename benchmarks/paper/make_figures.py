"""
Make all figures and the timing table for the paper from results/*.

Reads $FFSI_PAPER_DIR/results (default: benchmarks/paper/output) and writes the
figures to $FFSI_TEX_DIR/figures and the LaTeX tables/macros to
$FFSI_TEX_DIR/sections ($FFSI_TEX_DIR defaults to $FFSI_PAPER_DIR). Run from the repository root with the ffsi environment:
  python benchmarks/paper/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

import os
# results are read from $FFSI_PAPER_DIR; figures and LaTeX tables/macros are written
# to $FFSI_TEX_DIR (e.g. a clone of the paper's Overleaf project), which defaults to
# $FFSI_PAPER_DIR
RES_ROOT = Path(os.environ.get('FFSI_PAPER_DIR', Path(__file__).resolve().parent / 'output'))
ROOT = Path(os.environ.get('FFSI_TEX_DIR', RES_ROOT))
RES = RES_ROOT / 'results'
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / 'sections').mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (all-pairs safe) + neutral for the truth
C_FFSI = '#2a78d6'
C_FFSAS = '#eb6834'
C_FFSI_NOREG = '#1baf7a'
C_TRUTH = '#0b0b0b'
C_DATA = '#8a8984'
INK2 = '#52514e'

plt.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.5,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'savefig.bbox': 'tight', 'pdf.fonttype': 42,
})
W1, W2 = 3.3, 6.8  # single / double column widths (inches)


def load(code, name):
    """Return (summary dict, npz) for results/<code>/<name>."""
    d = json.load(open(RES / code / f'{name}.json'))
    z = np.load(RES / code / f'{name}.npz')
    return d, z


def problem(name):
    return np.load(RES / 'problems' / f'{name}.npz')


def save(fig, name):
    # no timestamp, so that unchanged figures give byte-identical PDFs
    fig.savefig(FIG / f'{name}.pdf', metadata={'CreationDate': None})
    # PNG previews go with the results, not into the LaTeX directory
    png = RES_ROOT / 'figures_png'
    png.mkdir(parents=True, exist_ok=True)
    fig.savefig(png / f'{name}.png', dpi=200)
    plt.close(fig)
    print('wrote', name)


def label_panels(axes):
    for k, ax in enumerate(np.ravel(axes)):
        ax.text(-0.02, 1.04, f'({chr(97 + k)})', transform=ax.transAxes,
                fontweight='bold', ha='right', va='bottom')


def vol_weighted(r, w):
    v = w * r ** 3
    return v / v.sum()


# --------------------------------------------------------------------------
def fig_sphere_benchmark():
    P = problem('sphere_benchmark')
    _, zr = load('ffsi', 'sphere_benchmark_gpu_reg')
    _, zn = load('ffsi', 'sphere_benchmark_gpu_noreg')
    _, zf = load('ffsas', 'sphere_benchmark_cuda')
    r, q = P['r'], P['q']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    ax[0].plot(r, zn['w0'], color=C_FFSI_NOREG, lw=0.9, label=r'FFSI ($\varsigma=0$)')
    ax[0].plot(r, zf['w0'], color=C_FFSAS, label='FFSAS')
    ax[0].plot(r, zr['w0'], color=C_FFSI, label=r'FFSI ($\varsigma=0.25$)')
    ax[0].plot(r, P['w_r_true'], color=C_TRUTH, ls='--', lw=1.0, label='Truth')
    ax[0].set_xlabel(r'$r$ (Å)')
    ax[0].set_ylabel(r'$w(r)$')
    ax[0].legend()
    ax[1].errorbar(q, P['I_data'], yerr=3 * P['I_data_std'], fmt='none',
                   ecolor=C_DATA, elinewidth=0.5, alpha=0.6, label=r'Data ($\pm3\sigma$)')
    ax[1].plot(q, zf['I_opt'], color=C_FFSAS, label='FFSAS')
    ax[1].plot(q, zr['I_opt'], color=C_FFSI, ls=':', label=r'FFSI ($\varsigma=0.25$)')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$q$ (Å$^{-1}$)')
    ax[1].set_ylabel(r'$I(q)$ (cm$^{-1}$)')
    ax[1].legend()
    label_panels(ax)
    save(fig, 'sphere_benchmark')


# --------------------------------------------------------------------------
CONVERGENCE = [
    # title, ffsi result, FFSAS result
    ('Sphere benchmark', 'sphere_benchmark_gpu_noreg', 'sphere_benchmark_cuda'),
    ('Challenge, $n_r=300$', 'sphere_challenge_300_gpu_noreg', 'sphere_challenge_300_cuda'),
    ('Cylinders (2D)', 'cylinder2d_gpu_noreg', 'cylinder2d_cuda'),
    ('Ellipsoids (2D)', 'ellipsoid2d_gpu_noreg', 'ellipsoid2d_cuda'),
    ('Cylinders (1D)', 'cylinder1d_gpu_noreg', 'cylinder1d_cuda'),
    ('Ellipsoids (1D)', 'ellipsoid1d_gpu_noreg', 'ellipsoid1d_cuda'),
    ('SANS', 'sphere_SANS_gpu_noreg', 'sphere_SANS_cuda'),
    ('SAXS', 'sphere_SAXS_gpu_noreg', 'sphere_SAXS_cuda'),
]


def fig_convergence():
    fig, axes = plt.subplots(2, 4, figsize=(W2, 3.8))
    for ax, (title, a, b) in zip(axes.ravel(), CONVERGENCE):
        try:
            _, za = load('ffsi', a)
            _, zb = load('ffsas', b)
        except FileNotFoundError:
            ax.set_visible(False)
            continue
        ha, hb = za['history'], zb['history']
        ax.plot(hb[:, 0], np.minimum.accumulate(hb[:, 1]), color=C_FFSAS, label='FFSAS')
        ax.plot(ha[:, 0], np.minimum.accumulate(ha[:, 1]), color=C_FFSI,
                marker='o', ms=3, label='FFSI')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('wall-clock time (s)')
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\chi^2$')
    axes[0, 0].legend()
    label_panels(axes)
    fig.tight_layout()
    save(fig, 'convergence')


# --------------------------------------------------------------------------
def fig_sphere_challenge():
    P = problem('sphere_challenge')
    res = [100, 200, 300, 400]
    fig, axes = plt.subplots(len(res), 2, figsize=(W2, 6.4), sharex='col')
    r_t = P['r']
    dens_t = P['w_r_true'] / (r_t[1] - r_t[0])
    for row, n in enumerate(res):
        r = P[f'r_{n}']
        dr = r[1] - r[0]
        ax = axes[row, 0]
        notes = []
        for code, name, col, lab in (('ffsas', f'sphere_challenge_{n}_cuda', C_FFSAS, 'FFSAS'),
                                     ('ffsi', f'sphere_challenge_{n}_gpu_noreg', C_FFSI, 'FFSI')):
            try:
                d, z = load(code, name)
            except FileNotFoundError:
                continue
            ax.plot(r, z['w0'] / dr, color=col, lw=0.9, label=lab)
            notes.append(rf'{lab}: $\chi^2$={d["chi2"]:.1e}, {d["solve_time"]:.1f} s')
            axes[row, 1].plot(P['q'], (z['I_opt'] - P['I_data']) / P['I_data_std'],
                              color=col, lw=0.7, label=lab)
        ax.plot(r_t, dens_t, color=C_TRUTH, ls='--', lw=0.8, label='Truth')
        ax.text(0.02, 0.95, '\n'.join(notes), transform=ax.transAxes, va='top',
                fontsize=6.5, color=INK2)
        ax.set_ylabel(r'$w(r)/\Delta r$ (Å$^{-1}$)')
        ax.set_title(rf'$n_r = {n}$')
        axes[row, 1].set_xscale('log')
        axes[row, 1].set_ylabel(r'$(I-\mu)/\sigma$')
    axes[-1, 0].set_xlabel(r'$r$ (Å)')
    axes[-1, 1].set_xlabel(r'$q$ (Å$^{-1}$)')
    label_panels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    save(fig, 'sphere_challenge')


# --------------------------------------------------------------------------
def fig_2d(name, keys, labels, units):
    P = problem(name)
    _, za = load('ffsi', f'{name}_gpu_noreg')
    try:
        _, zb = load('ffsas', f'{name}_cuda')
    except FileNotFoundError:
        zb = None
    fig = plt.figure(figsize=(W2, 3.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15])
    for k, (key, lab, unit) in enumerate(zip(keys, labels, units)):
        ax = fig.add_subplot(gs[0, k])
        x = P[key]
        if key in ('theta', 'phi'):
            x = np.rad2deg(x)
        ax.plot(x, P[f'w_{key}_true'], color=C_TRUTH, ls='--', lw=1.0, label='Truth')
        if zb is not None:
            ax.plot(x, zb[f'w{k}'], color=C_FFSAS, label='FFSAS')
        ax.plot(x, za[f'w{k}'], color=C_FFSI, ls=':', lw=1.6, label='FFSI')
        ax.set_xlabel(f'{lab} ({unit})')
        if k == 0:
            ax.set_ylabel('$w$')
            ax.legend()
    qx, qy = P['qx'], P['qy']
    ext = [qx.min(), qx.max(), qy.min(), qy.max()]
    I = P['I_data']
    panels = [('Data', np.log10(I), 'viridis', {})]
    lims = dict(vmin=-16, vmax=-4)
    panels.append(('FFSI misfit',
                   np.log10(np.abs(za['I_opt'] - I) / I + 1e-300), 'Greys', lims))
    if zb is not None:
        panels.append(('FFSAS misfit',
                       np.log10(np.abs(zb['I_opt'] - I) / I + 1e-300), 'Greys', lims))
    main_axes = list(fig.axes)
    for k, (title, img, cmap, kw) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        main_axes.append(ax)
        im = ax.imshow(img.T, origin='lower', extent=ext, cmap=cmap, **kw)
        ax.set_title(title)
        ax.set_xlabel(r'$q_x$ (Å$^{-1}$)')
        if k == 0:
            ax.set_ylabel(r'$q_y$ (Å$^{-1}$)')
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label(r'$\log_{10} I$' if k == 0 else r'$\log_{10}|I-\mu|/\mu$')
    label_panels(main_axes)
    fig.tight_layout()
    save(fig, name)


# --------------------------------------------------------------------------
def fig_real(dataset):
    P = problem(f'sphere_{dataset}')
    r, q = P['r'], P['q']
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.4))
    for code, name, col, lab, ls in (
            ('ffsi', f'sphere_{dataset}_gpu_noreg', C_FFSI_NOREG, r'FFSI ($\varsigma=0$)', '-'),
            ('ffsas', f'sphere_{dataset}_cuda', C_FFSAS, 'FFSAS', '-'),
            ('ffsi', f'sphere_{dataset}_gpu_reg', C_FFSI, None, '-')):
        try:
            d, z = load(code, name)
        except FileNotFoundError:
            continue
        if lab is None:
            lab = rf'FFSI ($\varsigma={d["sigma"]:g}$)'
        ax[0].plot(r, vol_weighted(r, z['w0']), color=col, lw=0.9, ls=ls, label=lab)
        ax[1].plot(q, z['I_opt'], color=col, lw=0.9, ls=ls,
                   label=rf'{lab}: $\chi^2/m$={d["chi2"] / len(q):.2f}')
    ax[0].set_xlabel(r'$r$ (Å)')
    ax[0].set_ylabel(r'volume-weighted $\hat w(r)$')
    ax[0].legend()
    ax[1].errorbar(q, P['I_data'], yerr=P['I_data_std'], fmt='o', ms=1.5,
                   color=C_DATA, ecolor='#c9c8c3', elinewidth=0.4, zorder=0, label='Data')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_ylim(bottom=max(1e-4, 0.3 * P['I_data'][P['I_data'] > 0].min()))
    ax[1].set_xlabel(r'$q$ (Å$^{-1}$)')
    ax[1].set_ylabel(r'$I(q)$ (cm$^{-1}$)')
    ax[1].legend()
    label_panels(ax)
    save(fig, f'sphere_{dataset}')


# --------------------------------------------------------------------------
def fig_1d_aniso():
    cases = [('cylinder1d', ['l', 'r'], ['$l$', '$r$'], 'Cylinders'),
             ('ellipsoid1d', ['rp', 're'], ['$r_p$', '$r_e$'], 'Ellipsoids')]
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.0))
    for row, (name, keys, labs, title) in enumerate(cases):
        P = problem(name)
        _, zn = load('ffsi', f'{name}_gpu_noreg')
        _, zf = load('ffsas', f'{name}_cuda')
        for k, (key, lab) in enumerate(zip(keys, labs)):
            ax = axes[row, k]
            x = P[key]
            ax.plot(x, P[f'w_{key}_true'], color=C_TRUTH, ls='--', lw=1.0, label='Truth')
            ax.plot(x, zf[f'w{k}'], color=C_FFSAS, label='FFSAS')
            ax.plot(x, zn[f'w{k}'], color=C_FFSI, ls=':', lw=1.6, label='FFSI')
            ax.set_xlabel(f'{lab} (Å)')
            ax.set_ylabel('$w$')
            ax.set_title(title)
        ax = axes[row, 2]
        q = P['q']
        ax.plot(q, P['I_data'], color=C_DATA, lw=2.5, label='Data')
        ax.plot(q, zf['I_opt'], color=C_FFSAS, label='FFSAS')
        ax.plot(q, zn['I_opt'], color=C_FFSI, ls=':', lw=1.6, label='FFSI')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$q$ (Å$^{-1}$)')
        ax.set_ylabel(r'$I(q)$ (cm$^{-1}$)')
        ax.set_title(title)
    label_panels(axes)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(axes[0, 2].get_lines()[0]); labels.append('Data')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, '1d_aniso')


# --------------------------------------------------------------------------
def fig_sans_scan():
    """Volume-weighted SANS distributions for a range of smoothing parameters."""
    P = problem('sphere_SANS')
    r = P['r']
    S = json.load(open(RES / 'smoothing_scan' / 'summary.json'))['sphere_SANS']
    keys = ['0', '0.003', '0.01', '0.03', '0.1', '0.3']
    cols = plt.cm.Blues(np.linspace(0.35, 1.0, len(keys)))
    fig, ax = plt.subplots(figsize=(W2, 2.6))
    for c, k in zip(cols, keys):
        w = np.load(RES / 'smoothing_scan' / f'sphere_SANS_vs{k}.npz')['w0']
        sig = S[k]['sigma'] or 0.0
        ax.plot(r, vol_weighted(r, w) / (r[1] - r[0]), color=c, lw=1.0,
                label=rf'$\varsigma={sig:.3g}$: $\chi^2/m={S[k]["chi2_red"]:.4f}$')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_xlabel(r'$r$ (Å)')
    ax.set_ylabel(r'volume density (Å$^{-1}$)')
    ax.legend(loc='upper left', ncol=2, fontsize=6.5)
    save(fig, 'sans_scan')


# --------------------------------------------------------------------------
TABLE = [
    ('Sphere benchmark', 'sphere_benchmark', ['gpu_reg', 'gpu_noreg', 'cpu_reg', 'cpu_noreg'], ['cuda', 'cpu']),
    ('Sphere challenge, $n_r=100$', 'sphere_challenge_100', ['gpu_noreg', 'gpu_reg'], ['cuda']),
    ('Sphere challenge, $n_r=200$', 'sphere_challenge_200', ['gpu_noreg', 'gpu_reg'], ['cuda']),
    ('Sphere challenge, $n_r=300$', 'sphere_challenge_300', ['gpu_noreg', 'gpu_reg'], ['cuda']),
    ('Sphere challenge, $n_r=400$', 'sphere_challenge_400', ['gpu_noreg', 'gpu_reg'], ['cuda']),
    ('Cylinders (2D)', 'cylinder2d', ['gpu_noreg'], ['cuda']),
    ('Ellipsoids (2D)', 'ellipsoid2d', ['gpu_noreg'], ['cuda']),
    ('Cylinders (1D)', 'cylinder1d', ['gpu_noreg', 'cpu_noreg'], ['cuda', 'cpu']),
    ('Ellipsoids (1D)', 'ellipsoid1d', ['gpu_noreg', 'cpu_noreg'], ['cuda', 'cpu']),
    ('SANS', 'sphere_SANS', ['gpu_reg', 'gpu_noreg', 'cpu_reg', 'cpu_noreg'], ['cuda', 'cpu']),
    ('SAXS', 'sphere_SAXS', ['gpu_reg', 'gpu_noreg', 'cpu_reg', 'cpu_noreg'], ['cuda', 'cpu']),
]


def summary_table():
    """Print a machine-readable summary and write a LaTeX table body."""
    rows = []
    out = {}
    for title, base, ffsi_tags, ffsas_tags in TABLE:
        for tag in ffsi_tags:
            try:
                d, _ = load('ffsi', f'{base}_{tag}')
            except FileNotFoundError:
                continue
            out[f'ffsi/{base}_{tag}'] = d
            dev, reg = tag.split('_')
            rows.append((title, 'FFSI', dev.upper(),
                         f'{d["sigma"]:g}' if reg == 'reg' else '0', d['chi2'], d['G_time'],
                         d['solve_time'], f'{d["n_evals"]} evals'))
        for tag in ffsas_tags:
            try:
                d, _ = load('ffsas', f'{base}_{tag}')
            except FileNotFoundError:
                continue
            out[f'ffsas/{base}_{tag}'] = d
            rows.append((title, 'FFSAS', 'GPU' if tag == 'cuda' else 'CPU', '--',
                         d['chi2'], d['G_time'] if d['G_time'] == d['G_time'] else float('nan'),
                         d['solve_time'], f'{d["nit"]} its'))
    json.dump(out, open(RES / 'summary.json', 'w'), indent=1, default=float)
    lines = []
    prev = None
    for title, code, dev, reg, c2, tg, ts, its in rows:
        if prev is not None and title != prev:
            lines.append(r'\midrule')
        t = title if title != prev else ''
        prev = title
        tgs = '--' if tg != tg else f'{tg:.2f}'
        lines.append(f'{t} & {code} & {dev} & {reg} & {c2:.1e} & {tgs} & {ts:.2f} & {its} \\\\')
    head = ['\\begin{tabular}{lllcrrrr}', '\\toprule',
            'Problem & Code & Device & $\\varsigma$ & $\\chi^2$ & $G$ & solve & iterations \\\\',
            '\\midrule']
    tail = ['\\bottomrule', '\\end{tabular}']
    (ROOT / 'sections' / 'table_timings.tex').write_text('\n'.join(head + lines + tail) + '\n')
    for r in rows:
        print('%-30s %-6s %-4s %-5s chi2=%9.2e G=%7.3f solve=%9.3f %s' % r)


CASE_NAMES = {'sphere_benchmark': 'SphBench', 'sphere_challenge_100': 'ChalA',
              'sphere_challenge_200': 'ChalB', 'sphere_challenge_300': 'ChalC',
              'sphere_challenge_400': 'ChalD', 'cylinder2d': 'CylTwo', 'ellipsoid2d': 'EllTwo',
              'cylinder1d': 'CylOne', 'ellipsoid1d': 'EllOne', 'sphere_SANS': 'SANS',
              'sphere_SAXS': 'SAXS'}
TAG_NAMES = {'gpu_reg': 'GpuReg', 'gpu_noreg': 'Gpu', 'cpu_reg': 'CpuReg',
             'cpu_noreg': 'Cpu', 'cuda': 'Gpu', 'cpu': 'Cpu'}


def tex_sci(x, digits=1):
    m, e = f'{x:.{digits}e}'.split('e')
    return f'{m}\\times10^{{{int(e)}}}'


def time_to(history, target):
    """First wall-clock time at which the history reaches chi^2 <= target."""
    idx = np.where(history[:, 1] <= target)[0]
    return float(history[idx[0], 0]) if len(idx) else float('nan')


def time_to_common(case):
    """Times (ffsi, FFSAS) on the GPU to reach the worse of the two final chi^2."""
    da, za = load('ffsi', f'{case}_gpu_noreg')
    db, zb = load('ffsas', f'{case}_cuda')
    target = max(da['chi2'], db['chi2']) * (1 + 1e-6)
    return time_to(za['history'], target), time_to(zb['history'], target), target


SPEEDUP_ROWS = [
    ('Sphere benchmark', 'sphere_benchmark'), ('Sphere challenge, $n_r=100$', 'sphere_challenge_100'),
    ('Sphere challenge, $n_r=200$', 'sphere_challenge_200'),
    ('Sphere challenge, $n_r=300$', 'sphere_challenge_300'),
    ('Sphere challenge, $n_r=400$', 'sphere_challenge_400'),
    ('SANS (polystyrene latex)', 'sphere_SANS'), ('SAXS (bimodal mixture)', 'sphere_SAXS'),
    ('Cylinders, 1D', 'cylinder1d'), ('Ellipsoids, 1D', 'ellipsoid1d'),
    ('Cylinders, 2D', 'cylinder2d'), ('Ellipsoids, 2D', 'ellipsoid2d'),
]


def speedup_table():
    """LaTeX rows: problem, N, m, n, target chi^2, times to target, speed-up."""
    lines = []
    for title, case in SPEEDUP_ROWS:
        try:
            t_a, t_b, target = time_to_common(case)
        except FileNotFoundError:
            continue
        _, za = load('ffsi', f'{case}_gpu_noreg')
        npar = len([k for k in za.files if k.startswith('w')])
        nvar = sum(za[f'w{k}'].size for k in range(npar))
        m = int(np.load(RES / 'ffsi' / f'{case}_gpu_noreg.npz')['I_opt'].size)
        fmt = lambda t: f'{t:.2g}' if t < 10 else f'{t:.0f}'
        lines.append(f'{title} & {npar} & {m} & {nvar} & ${tex_sci(target).replace(chr(92)*2, chr(92))}$ '
                     f'& {fmt(t_a)} & {fmt(t_b)} & {fmt(t_b / t_a)} \\\\')
    head = ['\\begin{tabular}{lrrrcrrr}', '\\toprule',
            'Problem & $N$ & $m$ & $\\sum_k n_k$ & target $\\chi^2$ & \\ffsi & \\ffsas & speed-up \\\\',
            '\\midrule']
    tail = ['\\bottomrule', '\\end{tabular}']
    (ROOT / 'sections' / 'table_speedup.tex').write_text('\n'.join(head + lines + tail) + '\n')
    print('wrote table_speedup.tex')


def numbers_tex():
    """Write LaTeX macros for every timing/chi^2 so the text never drifts."""
    lines = ['% generated by make_figures.py -- do not edit']
    for code in ('ffsi', 'ffsas'):
        for f in sorted((RES / code).glob('*.json')):
            name = f.stem
            for case, cname in sorted(CASE_NAMES.items(), key=lambda kv: -len(kv[0])):
                if name.startswith(case + '_'):
                    tag = name[len(case) + 1:]
                    break
            else:
                continue
            if tag not in TAG_NAMES:
                continue
            d = json.load(open(f))
            pre = ('ffsi' if code == 'ffsi' else 'ffsas') + cname + TAG_NAMES[tag]
            t = d['solve_time']
            lines.append(f'\\newcommand{{\\{pre}Time}}{{{t:.2g}}}' if t < 10 else
                         f'\\newcommand{{\\{pre}Time}}{{{t:.0f}}}')
            lines.append(f'\\newcommand{{\\{pre}Chi}}{{{tex_sci(d["chi2"])}}}')
    # time for each code to reach a common target chi^2 (the worse final value)
    for case, cname in CASE_NAMES.items():
        try:
            t_a, t_b, target = time_to_common(case)
        except FileNotFoundError:
            continue
        for pre, t in (('ffsi', t_a), ('ffsas', t_b)):
            lines.append(f'\\newcommand{{\\{pre}{cname}Target}}{{{t:.2g}}}' if t < 10 else
                         f'\\newcommand{{\\{pre}{cname}Target}}{{{t:.0f}}}')
        r = t_b / t_a
        lines.append(f'\\newcommand{{\\speedupTarget{cname}}}{{{r:.2g}}}' if r < 10 else
                     f'\\newcommand{{\\speedupTarget{cname}}}{{{r:.0f}}}')
        lines.append(f'\\newcommand{{\\target{cname}}}{{{tex_sci(target)}}}')
    # speed-ups (FFSAS GPU solve time / ffsi GPU solve time, unregularized)
    for case, cname in CASE_NAMES.items():
        try:
            a = json.load(open(RES / 'ffsi' / f'{case}_gpu_noreg.json'))['solve_time']
            b = json.load(open(RES / 'ffsas' / f'{case}_cuda.json'))['solve_time']
        except FileNotFoundError:
            continue
        r = b / a
        lines.append(f'\\newcommand{{\\speedup{cname}}}{{{r:.2g}}}' if r < 10 else
                     f'\\newcommand{{\\speedup{cname}}}{{{r:.0f}}}')
    (ROOT / 'sections' / 'numbers.tex').write_text('\n'.join(lines) + '\n')
    print('wrote numbers.tex')


if __name__ == '__main__':
    for f in (fig_sphere_benchmark, fig_convergence, fig_sphere_challenge,
              lambda: fig_2d('cylinder2d', ['l', 'r', 'theta', 'phi'],
                             ['$l$', '$r$', r'$\theta$', r'$\phi$'], ['Å', 'Å', 'deg', 'deg']),
              lambda: fig_2d('ellipsoid2d', ['rp', 're', 'theta', 'phi'],
                             ['$r_p$', '$r_e$', r'$\theta$', r'$\phi$'], ['Å', 'Å', 'deg', 'deg']),
              fig_1d_aniso, fig_sans_scan, lambda: fig_real('SANS'), lambda: fig_real('SAXS'), summary_table,
              numbers_tex, speedup_table):
        try:
            f()
        except FileNotFoundError as err:
            print('skipping (missing results):', err)
