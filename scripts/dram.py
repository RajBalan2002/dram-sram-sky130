"""1T1C DRAM column-slice netlist builder (folded bitline).

  cell:   access NMOS (bl - wl - sn) + Cs from sn to plate (VCP = VDD/2)
  BL/BLB: lumped Cbl each (BLB = reference bitline of the folded pair)
  PRE/EQ: 3 NMOS on PEQ, bitlines -> VBLP = VDD/2
  SA:     cross-coupled latch, NMOS sources on SAN, PMOS sources on SAP
  column: CSL NMOS pass gates bl<->io, blb<->iob (write driver = ideal io sources)

All times are in ns. Control signals are PWL sources built from event lists.
"""

from spice import fet

DEFAULT = dict(
    vdd=1.8, vpp=2.6, cs=20e-15, cbl=100e-15,
    w_acc=0.42, l_acc=0.15, vwl_low=0.0,
    w_pre=1.0, w_san=1.0, w_sap=2.0, w_csl=2.0,
    peq_high="vpp",          # 'vpp' (fixed) or 'vdd' (original, too weak)
    sa_vt_n="svt",           # 'svt' or 'lvt' sense-amp NMOS pair
    sa_vt_p="svt",           # 'svt' or 'lvt' sense-amp PMOS pair
    l_sap=0.15,              # PMOS latch length (lvt PMOS needs >= 0.35)
)

SA_LVT = dict(sa_vt_n="lvt", sa_vt_p="lvt", l_sap=0.35, w_sap=4.0)   # all low-Vt latch
SA_2X = dict(w_san=2.0, w_sap=4.0)                                    # standard-Vt, 2x wider
SA_OPTIONS = {"svt": {}, "svt2x": SA_2X, "lvt": SA_LVT}


def pwl(points, unit=1e-9):
    """points: [(t_ns, v), ...] -> 'PWL(...)'"""
    return "PWL(" + " ".join(f"{t*unit:.4g} {v:.4g}" for t, v in points) + ")"


def edge(t, v0, v1, tr=0.1):
    return [(t, v0), (t + tr, v1)]


def read_timing(p, t0=0.0):
    """One ACT -> sense -> restore -> PRE cycle starting at t0 (ns).
    Returns dict of event times used for measurements."""
    return dict(pre_off=t0 + 2.0, wl_on=t0 + 3.0, san=t0 + 5.0, sap=t0 + 5.5,
                wl_off=t0 + 9.0, sa_off=t0 + 10.0, pre_on=t0 + 10.0, end=t0 + 12.0)


def column(p, cycles, write=None, dvt=None, extra=""):
    """Build the netlist body.

    cycles: list of timing dicts (read_timing). write: dict(cycle=i, data=0/1,
    t_on, t_off) drives io/iob and pulses CSL during that cycle.
    dvt: optional dict of per-device Vt shifts (Monte Carlo).
    """
    q = dict(DEFAULT)
    q.update(p)
    vdd, vpp = q["vdd"], q["vpp"]
    vblp = vdd / 2
    peq_hi = vpp if q["peq_high"] == "vpp" else vdd
    dvt = dvt or {}

    # control waveforms
    peq, wl, san, sap = [(0, peq_hi)], [(0, q["vwl_low"])], [(0, vblp)], [(0, vblp)]
    for c in cycles:
        peq += edge(c["pre_off"], peq_hi, 0) + edge(c["pre_on"], 0, peq_hi)
        wl += edge(c["wl_on"], q["vwl_low"], vpp, 0.2) + edge(c["wl_off"], vpp, q["vwl_low"], 0.2)
        san += edge(c["san"], vblp, 0, 0.5) + edge(c["sa_off"], 0, vblp)
        sap += edge(c["sap"], vblp, vdd, 0.5) + edge(c["sa_off"], vdd, vblp)
    csl = [(0, 0)]
    io_v, iob_v = vblp, vblp
    if write:
        csl += edge(write["t_on"], 0, vdd) + edge(write["t_off"], vdd, 0)
        io_v, iob_v = (vdd, 0.0) if write["data"] else (0.0, vdd)

    b = []
    b.append(f".param vdd={vdd} vblp={vblp} vpp={vpp}\n")
    b.append(f"VDD vdd 0 {vdd}\nVBLP vblp 0 {vblp}\nVCP vcp 0 {vblp}\n")
    b.append(f"VPEQ peq 0 {pwl(peq)}\nVWL wl 0 {pwl(wl)}\n")
    b.append(f"VSAN san 0 {pwl(san)}\nVSAP sap 0 {pwl(sap)}\n")
    b.append(f"VCSL csl 0 {pwl(csl)}\nVIO io 0 {io_v}\nVIOB iob 0 {iob_v}\n")
    # cell
    b.append(fet("ma", "bl", "wl", "sn", "0", "n", q["w_acc"], q["l_acc"], dvt.get("ma", 0)))
    b.append(f"CS sn vcp {q['cs']:.4g}\n")
    # bitlines
    b.append(f"CBL bl 0 {q['cbl']:.4g}\nCBLB blb 0 {q['cbl']:.4g}\n")
    # precharge / equalize
    b.append(fet("mpr1", "bl", "peq", "vblp", "0", "n", q["w_pre"], 0.15))
    b.append(fet("mpr2", "blb", "peq", "vblp", "0", "n", q["w_pre"], 0.15))
    b.append(fet("meq", "bl", "peq", "blb", "0", "n", q["w_pre"], 0.15))
    # sense amp
    vn, vp, lp = q["sa_vt_n"], q["sa_vt_p"], q["l_sap"]
    b.append(fet("mn1", "bl", "blb", "san", "0", "n", q["w_san"], 0.15, dvt.get("mn1", 0), vn))
    b.append(fet("mn2", "blb", "bl", "san", "0", "n", q["w_san"], 0.15, dvt.get("mn2", 0), vn))
    b.append(fet("mp1", "bl", "blb", "sap", "vdd", "p", q["w_sap"], lp, dvt.get("mp1", 0), vp))
    b.append(fet("mp2", "blb", "bl", "sap", "vdd", "p", q["w_sap"], lp, dvt.get("mp2", 0), vp))
    # column select / write path
    b.append(fet("mcs1", "bl", "csl", "io", "0", "n", q["w_csl"], 0.15))
    b.append(fet("mcs2", "blb", "csl", "iob", "0", "n", q["w_csl"], 0.15))
    b.append(extra)
    return "".join(b)


def read_measures(c, tag="", vdd=1.8):
    """.meas lines for one read cycle c (timing dict). Requires vectors vdiff/avdiff."""
    s = tag
    v = (f"meas tran vsn_before{s} find v(sn) at={c['wl_on'] - 0.1}n\n"
         f"meas tran dv_share{s} find vdiff at={c['san'] - 0.1}n\n"
         f"meas tran tsense{s} trig at={c['san']}n targ avdiff val={0.8 * vdd:.4g} rise=1 td={c['san']}n\n"
         f"meas tran vsn_restored{s} find v(sn) at={c['wl_off'] - 0.1}n\n"
         f"meas tran vsn_hold{s} find v(sn) at={c['end'] - 0.1}n\n"
         f"meas tran tprecharge{s} trig at={c['pre_on'] + 0.1}n targ avdiff val=0.01 fall=1 td={c['pre_on']}n\n")
    return v
