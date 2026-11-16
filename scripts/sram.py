"""6T SRAM cell netlist pieces and static-noise-margin math.

Cell (one side):   PU  pfet  out <- vdd   (gate = inp)
                   PD  nfet  out -> gnd   (gate = inp)
                   PG  nfet  bl <-> out   (gate = wl)
Default sizing (W/L in um): PD 0.84/0.15, PG 0.42/0.15, PU 0.42/0.15
  -> cell ratio beta = PD/PG = 2 (read stability), pull-up ratio PU/PG = 1 (pfet is ~2.5x weaker)
"""
import numpy as np
from spice import fet

SIZES = dict(pd=(0.84, 0.15), pg=(0.42, 0.15), pu=(0.42, 0.15))
TX = [("pu", "p"), ("pd", "n"), ("pg", "n")]


def half(tag, inp, out, wl, bl, dvt=None, sizes=SIZES):
    """Inverter + access transistor; dvt = {'pu':..,'pd':..,'pg':..}."""
    dvt = dvt or {}
    s = sizes
    return (fet(f"pu{tag}", out, inp, "vdd", "vdd", "p", *s["pu"], dvt.get("pu", 0))
            + fet(f"pd{tag}", out, inp, "0", "0", "n", *s["pd"], dvt.get("pd", 0))
            + fet(f"pg{tag}", bl, wl, out, "0", "n", *s["pg"], dvt.get("pg", 0)))


def butterfly_body(name, cells, vdd=1.8, step=0.005):
    """Open-loop VTCs for hold (WL=0) and read (WL=VDD, BL=VDD) in one DC sweep.

    cells: list of dicts {'a': dvt_halfA, 'b': dvt_halfB}. Every half-cell's
    input is the swept node vin, so N cells cost one sweep.
    Saved vectors per cell i: hold A, hold B, read A, read B outputs.
    """
    b = [f"VDD vdd 0 {vdd}\nVIN vin 0 0\nVWLH wlh 0 0\nVWLR wlr 0 {vdd}\nVBLR blr 0 {vdd}\n"]
    vecs = []
    for i, c in enumerate(cells):
        b.append(half(f"ha{i}", "vin", f"oha{i}", "wlh", "blr", c["a"]))
        b.append(half(f"hb{i}", "vin", f"ohb{i}", "wlh", "blr", c["b"]))
        b.append(half(f"ra{i}", "vin", f"ora{i}", "wlr", "blr", c["a"]))
        b.append(half(f"rb{i}", "vin", f"orb{i}", "wlr", "blr", c["b"]))
        vecs += [f"v(oha{i})", f"v(ohb{i})", f"v(ora{i})", f"v(orb{i})"]
    b.append(f".control\ndc VIN 0 {vdd} {step}\nwrdata {name}.dat {' '.join(vecs)}\n.endc\n")
    return "".join(b)


def write_body(name, cells, vdd=1.8, step=0.005):
    """Write margin: cell holds Q=1, WL=BLB=VDD, sweep BL from VDD down.
    WM = BL voltage at which Q flips (bigger = easier to write)."""
    b = [f"VDD vdd 0 {vdd}\nVWL wl 0 {vdd}\nVBL bl 0 {vdd}\nVBLB blb 0 {vdd}\n"]
    vecs, ns = [], []
    for i, c in enumerate(cells):
        b.append(half(f"a{i}", f"qb{i}", f"q{i}", "wl", "bl", c["a"]))
        b.append(half(f"b{i}", f"q{i}", f"qb{i}", "wl", "blb", c["b"]))
        vecs.append(f"v(q{i})")
        ns.append(f"v(q{i})={vdd} v(qb{i})=0")
    b.append(".nodeset " + " ".join(ns) + "\n")
    b.append(f".control\ndc VBL {vdd} 0 -{step}\nwrdata {name}.dat {' '.join(vecs)}\n.endc\n")
    return "".join(b)


def write_body_tran(name, cells, vdd=1.8, tramp=1e-6):
    """Write margin for Monte Carlo: same test as write_body, but BL is ramped
    slowly in a transient. Many cells can share one netlist because each cell
    keeps its own state (a DC sweep can lose state when the solver restarts).
    .meas tf<i> = time Q crosses VDD/2; WM = BL voltage at that time."""
    b = [f"VDD vdd 0 {vdd}\nVWL wl 0 {vdd}\nVBLB blb 0 {vdd}\n"
         f"VBL bl 0 PWL(0 {vdd} 1n {vdd} {1e-9 + tramp:.4g} 0)\n"]
    ic, meas = [], []
    for i, c in enumerate(cells):
        b.append(half(f"a{i}", f"qb{i}", f"q{i}", "wl", "bl", c["a"]))
        b.append(half(f"b{i}", f"q{i}", f"qb{i}", "wl", "blb", c["b"]))
        ic.append(f"v(q{i})={vdd} v(qb{i})=0")
        meas.append(f"meas tran tf{i} when v(q{i})={vdd/2} fall=1")
    b.append(".ic " + " ".join(ic) + "\n")
    b.append(f".control\ntran {tramp/2000:.4g} {1e-9 + tramp:.4g}\n" + "\n".join(meas) + "\n.endc\n")
    return "".join(b)


def wm_from_time(tf, vdd=1.8, tramp=1e-6):
    if tf != tf:            # nan: never flipped
        return 0.0
    return vdd * (1 - (tf - 1e-9) / tramp)


def read_current_body(vdd=1.8):
    """Cell read current: Q=0, WL=BL=BLB=VDD; current drawn from BL."""
    return (f"VDD vdd 0 {vdd}\nVWL wl 0 {vdd}\nVBL bl 0 {vdd}\nVBLB blb 0 {vdd}\n"
            + half("a", "qb", "q", "wl", "bl") + half("b", "q", "qb", "wl", "blb")
            + f".nodeset v(q)=0 v(qb)={vdd}\n"
            + ".control\nop\nlet iread = -i(VBL)\nprint iread\n.endc\n")


def snm(x, ya, yb):
    """Seevinck static noise margin from two open-loop VTCs.

    Curve 1: Q = A(QB)  -> points (x, ya).   Curve 2: QB = B(Q) -> points (yb, x).
    Rotate by 45 deg; the largest embedded square in each eye has side
    max(vertical gap)/sqrt(2). SNM = smaller of the two eyes.
    """
    r = np.sqrt(2)
    u1, v1 = (x - ya) / r, (x + ya) / r
    u2, v2 = (yb - x) / r, (yb + x) / r
    o1, o2 = np.argsort(u1), np.argsort(u2)
    u = np.linspace(max(u1.min(), u2.min()), min(u1.max(), u2.max()), 2000)
    d = np.interp(u, u1[o1], v1[o1]) - np.interp(u, u2[o2], v2[o2])
    e1, e2 = d.max(), (-d).max()
    return min(e1, e2) / r, (e1 / r, e2 / r)


def write_margin(vbl, q, vdd=1.8):
    i = np.where(q < vdd / 2)[0]
    if len(i) == 0:
        return 0.0          # never flipped: cannot be written
    k = i[0]
    return float(np.interp(vdd / 2, [q[k], q[k - 1]], [vbl[k], vbl[k - 1]]))
