"""Small ngspice driver used by every experiment in this repo.

- builds a netlist from a body string, adds the SKY130 .lib line + temperature
- runs `ngspice -b` in sim/ (where .spiceinit lives)
- parses `.meas` results into a dict (failed measurements -> nan)
- loads `wrdata` files into numpy arrays
- runs many netlists in parallel (corners, sweeps)

Model library: $SKY130_LIB if set, else $PDK_ROOT/$PDK/libs.tech/ngspice/sky130.lib.spice
"""
import os
import re
import subprocess
import concurrent.futures as cf
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIM = os.path.join(ROOT, "sim")
PLOTS = os.path.join(ROOT, "plots")
RESULTS = os.path.join(ROOT, "results")
for _d in (SIM, PLOTS, RESULTS):
    os.makedirs(_d, exist_ok=True)

SPICEINIT = "set ngbehavior=hsa\nset ng_nomodcheck\noption klu\n"
with open(os.path.join(SIM, ".spiceinit"), "w") as _f:
    _f.write(SPICEINIT)

# SKY130 corner names -> what they actually do to the devices.
# Verified with an Id measurement (see README): in this PDK "sf" = fast NMOS / slow PMOS.
CORNERS = ["tt", "ss", "ff", "sf", "fs"]

# Pelgrom Vt-mismatch coefficients taken from the PDK mismatch files
# (sky130_fd_pr__{n,p}fet_01v8__vth0_slope), units V*um:  sigma(Vt) = A / sqrt(W*L)
AVT = {("n", "svt"): 3.356e-3, ("p", "svt"): 5.856e-3,
       ("n", "lvt"): 5.456e-3, ("p", "lvt"): 13.89e-3}


def corner_labels():
    """Corner tick labels that say what each corner does (from results/corner_check.csv)."""
    lab = {c: c for c in CORNERS}
    p = os.path.join(RESULTS, "corner_check.csv")
    if os.path.exists(p):
        import csv
        for r in csv.DictReader(open(p)):
            if r["corner"] in ("sf", "fs"):
                lab[r["corner"]] = f"{r['corner']}\n(N {r['nmos']}, P {r['pmos']})"
    return lab


def model_lib():
    lib = os.environ.get("SKY130_LIB")
    if lib:
        return lib
    root, pdk = os.environ.get("PDK_ROOT"), os.environ.get("PDK", "sky130A")
    if not root:
        raise SystemExit("Set PDK_ROOT (e.g. export PDK_ROOT=$HOME/.ciel PDK=sky130A)")
    return f"{root}/{pdk}/libs.tech/ngspice/sky130.lib.spice"


def fet(name, d, g, s, b, kind, w, l, dvt=0.0, vt="svt"):
    """One SKY130 1.8 V transistor. kind = 'n' or 'p'.

    vt  = 'svt' (standard) or 'lvt' (low threshold).
    dvt = threshold shift in volts (Monte Carlo mismatch). It is applied as a
    series gate source, which is equivalent to a Vt shift for DC and transient.
    Junction area/perimeter use a 0.29 um diffusion length so drain/source
    junction capacitance and leakage are included.
    """
    model = "sky130_fd_pr__nfet_01v8" if kind == "n" else "sky130_fd_pr__pfet_01v8"
    if vt == "lvt":   # low-Vt flavour (note: pfet_01v8_lvt needs L >= 0.35 um)
        model += "_lvt"
    ad = round(w * 0.29, 4)
    pd = round(2 * (w + 0.29), 4)
    geo = f"W={w} L={l} nf=1 ad={ad} as={ad} pd={pd} ps={pd}"
    if dvt == 0.0:
        return f"X{name} {d} {g} {s} {b} {model} {geo}\n"
    gi = f"g_{name}"
    # NMOS: higher Vt == lower effective gate voltage. PMOS: higher |Vt| == higher gate.
    src = (f"V{name}mm {g} {gi} {dvt:.6g}\n" if kind == "n"
           else f"V{name}mm {gi} {g} {dvt:.6g}\n")
    return src + f"X{name} {d} {gi} {s} {b} {model} {geo}\n"


def sigma_vt(kind, w, l, vt="svt"):
    return AVT[(kind, vt)] / np.sqrt(w * l)


_MEAS = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*([-+]?\d[\d.]*(?:[eE][-+]?\d+)?)")
_FAIL = re.compile(r"^\s*meas\s+\w+\s+([A-Za-z_]\w*)\b.*failed", re.I)


def run(name, body, corner="tt", temp=27, timeout=1800):
    """Write sim/<name>.spice, run it, return dict of .meas results."""
    text = (f"* {name}\n.lib {model_lib()} {corner}\n.temp {temp}\n"
            f".option temp={temp}\n{body}\n.end\n")
    path = os.path.join(SIM, f"{name}.spice")
    with open(path, "w") as f:
        f.write(text)
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_WAIT_POLICY="passive")   # avoid thread oversubscription in parallel runs
    p = subprocess.run(["ngspice", "-b", f"{name}.spice"], cwd=SIM, env=env,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=timeout)
    out = p.stdout + p.stderr
    with open(os.path.join(SIM, f"{name}.log"), "w") as f:
        f.write(out)
    res = {}
    for line in out.splitlines():
        m = _MEAS.match(line)
        if m:
            res[m.group(1).lower()] = float(m.group(2))
            continue
        m = _FAIL.match(line)
        if m:
            res[m.group(1).lower()] = float("nan")
    if "rror" in out and not res and "wrdata" not in body:
        raise RuntimeError(f"ngspice error in {name}, see sim/{name}.log")
    return res


def run_many(jobs, workers=None):
    """jobs: list of dicts with keys for run(). Returns results in the same order."""
    workers = workers or max(1, os.cpu_count() or 1)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, **j) for j in jobs]
        return [f.result() for f in futs]


def load(name):
    """Load sim/<name>.dat written by `wrdata <name>.dat v1 v2 ...`.

    Returns (x, Y) where Y[:, k] is the k-th saved vector."""
    d = np.loadtxt(os.path.join(SIM, f"{name}.dat"))
    if d.ndim == 1:
        d = d[None, :]
    return d[:, 0], d[:, 1::2]


# ---------- plotting style (validated categorical palette, light mode) ----------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "lines.linewidth": 2,
        "legend.frameon": False, "font.size": 10, "axes.titlesize": 11,
    })
    return plt


def save(fig, name):
    p = os.path.join(PLOTS, name)
    fig.savefig(p)
    print("wrote", os.path.relpath(p, ROOT))
