"""M2: DRAM sweeps.

  (a) bitline signal vs storage capacitance (simulation vs charge-sharing theory)
  (b) retention time vs temperature from cell leakage (4 access-device options)
  (c) PVT corners: process x temperature x VDD for read 1 / read 0

Outputs: results/m2_*.csv, results/corner_check.csv, plots/m2_*.png
"""
import csv
import numpy as np
import spice
from m1_dram import read_job
from dram import SA_OPTIONS

TEMPS = [-40, 27, 125]
VDDS = [1.62, 1.8, 1.98]


# ------------------------------------------------------------------ helpers
def corner_check():
    """Measure NMOS/PMOS drive in each corner so plots are labelled by what the
    corner really does (SKY130's 'sf'/'fs' naming is easy to get backwards)."""
    body = ("Vg g 0 1.8\nVd d 0 1.8\nVdd vdd 0 1.8\nVgp gp 0 0\nVdp dp 0 0\n"
            + spice.fet("n", "d", "g", "0", "0", "n", 1, 0.15)
            + spice.fet("p", "dp", "gp", "vdd", "vdd", "p", 1, 0.15)
            + ".control\nop\nlet idn = -i(Vd)\nlet idp = i(Vdp)\n"
              "print idn idp\n.endc\n")
    out = {}
    for c in spice.CORNERS:
        spice.run(f"m2_corner_{c}", body, corner=c)
        log = open(f"{spice.SIM}/m2_corner_{c}.log").read()
        vals = {}
        for line in log.splitlines():
            for k in ("idn", "idp"):
                if line.strip().startswith(k):
                    vals[k] = float(line.split("=")[1])
        out[c] = vals
    tt = out["tt"]
    labels = {}
    with open(f"{spice.RESULTS}/corner_check.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corner", "Idsat_n_uA_per_um", "Idsat_p_uA_per_um", "nmos", "pmos"])
        for c, v in out.items():
            fn = "fast" if v["idn"] > tt["idn"] * 1.03 else "slow" if v["idn"] < tt["idn"] * 0.97 else "typ"
            fp = "fast" if v["idp"] > tt["idp"] * 1.03 else "slow" if v["idp"] < tt["idp"] * 0.97 else "typ"
            w.writerow([c, round(v["idn"] * 1e6, 1), round(v["idp"] * 1e6, 1), fn, fp])
            labels[c] = c if c in ("tt", "ss", "ff") else f"{c}\n(N {fn}, P {fp})"
    return labels


# ------------------------------------------------------------- (a) dV vs Cs
def dv_vs_cs(plt):
    cs_list = [5, 10, 15, 20, 25, 30, 40, 50]
    jobs = []
    for cs in cs_list:
        jobs += [read_job(f"m2_cs{cs}_r1", 1.8, {"cs": cs * 1e-15}),
                 read_job(f"m2_cs{cs}_r0", 0.0, {"cs": cs * 1e-15})]
    res = spice.run_many(jobs)
    rows = []
    for i, cs in enumerate(cs_list):
        d1, d0 = res[2 * i]["dv_share"], res[2 * i + 1]["dv_share"]
        rows.append([cs, d1, d0, (d1 - d0) / 2, (d1 + d0) / 2, 0.9 * cs / (cs + 100)])
    rows = np.array(rows)
    # fit effective bitline capacitance: sym = 0.89 * Cs / (Cs + Cbl_eff)
    swing = 0.885  # bitline sits ~15 mV below 0.9 V after the PEQ edge (charge injection)
    cbl_eff = np.median(rows[:, 0] * (swing / rows[:, 3] - 1))
    with open(f"{spice.RESULTS}/m2_dv_vs_cs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cs_fF", "dv_read1", "dv_read0", "dv_symmetric", "dv_offset", "dv_theory_cbl100"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.linspace(4, 52, 100)
    ax.plot(x, 900 * x / (x + 100), color=spice.INK2, ls=":", lw=1.2,
            label="theory, Cbl = 100 fF")
    ax.plot(rows[:, 0], rows[:, 1] * 1e3, marker="o", ms=5, label="read 1 (simulated)")
    ax.plot(rows[:, 0], -rows[:, 2] * 1e3, marker="s", ms=5, label="read 0 (simulated, |ΔV|)")
    ax.set_xlabel("Storage capacitance Cs (fF)")
    ax.set_ylabel("Bitline signal |ΔV| (mV)")
    ax.set_title(f"Charge sharing: effective Cbl ≈ {cbl_eff:.0f} fF, "
                 f"WL-coupling offset ≈ {np.mean(rows[:, 4])*1e3:+.0f} mV")
    ax.legend(loc="upper left")
    spice.save(fig, "m2_dv_vs_cs.png")
    plt.close(fig)
    return dict(cbl_eff=cbl_eff, offset=float(np.mean(rows[:, 4])))


# ------------------------------------------------------------ (b) retention
VARIANTS = [  # label, L_access, WL low level
    ("WL low = 0 V", 0.15, 0.0),
    ("WL low = −0.3 V (negative wordline)", 0.15, -0.3),
]
CONDITIONS = [  # label, stored data, bitline voltage while the cell waits
    ("'1', BL at VDD/2 (standby)", 1, 0.9),
    ("'0', BL at VDD/2 (standby)", 0, 0.9),
    ("'1', BL held at 0 (disturb)", 1, 0.0),
    ("'0', BL held at VDD (disturb)", 0, 1.8),
]
RET_TEMPS = [27, 55, 85, 105, 125]   # below ~27 C leakage reaches the fA noise floor
CS = 20e-15
VFAIL_DV = 0.050        # read fails when bitline signal drops below 50 mV
_trapz = getattr(np, "trapezoid", None) or np.trapz


def leakage_job(name, l_acc, vwl, vbl, data):
    """DC sweep of the storage node; gmin lowered so fA leakage is not swamped."""
    sweep = "dc VSN 1.8 0.6 -0.01" if data else "dc VSN 0 0.8 0.01"
    return (".option gmin=1e-18\n"
            f"VSN sn 0 {1.8 if data else 0}\nVBL bl 0 {vbl}\nVWL wl 0 {vwl}\nVB body 0 0\n"
            + spice.fet("ma", "bl", "wl", "sn", "body", "n", 0.42, l_acc)
            + f".control\n{sweep}\n"
              "let ileak = abs(i(VSN))\nlet ich = abs(i(VBL))\nlet ibody = abs(i(VB))\n"
              f"wrdata {name}.dat ileak ich ibody\n.endc\n")


def tret(v, ileak, data, vfail):
    if data:
        m = v >= vfail
        return _trapz(CS / ileak[m][::-1], v[m][::-1])
    m = v <= vfail
    return _trapz(CS / ileak[m], v[m])


def retention(plt, ratio):
    vfail1 = 0.9 + VFAIL_DV / ratio
    vfail0 = 0.9 - VFAIL_DV / ratio
    jobs, keys = [], []
    for k, (vlab, l, vwl) in enumerate(VARIANTS):
        for ci, (clab, data, vbl) in enumerate(CONDITIONS):
            for t in RET_TEMPS:
                nm = f"m2_leak_v{k}_c{ci}_t{t}".replace("-", "m")
                jobs.append(dict(name=nm, body=leakage_job(nm, l, vwl, vbl, data), temp=t))
                keys.append((k, ci, t, nm))
    spice.run_many(jobs)
    table, rows = {}, []
    for k, ci, t, nm in keys:
        v, Y = spice.load(nm)
        data = CONDITIONS[ci][1]
        tr = tret(v, Y[:, 0], data, vfail1 if data else vfail0)
        table[(k, ci, t)] = tr
        rows.append([VARIANTS[k][0], CONDITIONS[ci][0], t, Y[0, 0], Y[0, 1], Y[0, 2], tr])
    with open(f"{spice.RESULTS}/m2_retention.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device", "condition", "temp_C", "I_leak_A_start", "I_channel_A",
                    "I_body_A", "retention_s"])
        w.writerows(rows)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    for ci, (clab, data, vbl) in enumerate(CONDITIONS):
        a1.semilogy(RET_TEMPS, [table[(0, ci, t)] for t in RET_TEMPS], marker="o", ms=5,
                    label=clab)
    a1.set_title("Access NMOS W/L 0.42/0.15 µm, WL low = 0 V")
    a1.set_ylabel(f"Retention time (s)  [fail: ΔV < {VFAIL_DV*1e3:.0f} mV]")
    a2.semilogy(RET_TEMPS, [table[(0, 0, t)] for t in RET_TEMPS], color=spice.INK2, lw=6,
                alpha=0.18, label="junction-leakage floor ('1', BL at VDD/2)")
    for k, (vlab, l, vwl) in enumerate(VARIANTS):
        a2.semilogy(RET_TEMPS, [table[(k, 2, t)] for t in RET_TEMPS], marker="os"[k], ms=5,
                    color=spice.SERIES[k], label=f"'1', BL held at 0, {vlab}")
    a2.set_title("Negative wordline vs channel leakage ('1')")
    for a in (a1, a2):
        a.axhline(64e-3, color=spice.INK2, lw=0.9, ls="--")
        a.annotate("64 ms", (125, 64e-3), xytext=(-28, 4), textcoords="offset points",
                   fontsize=8, color=spice.INK2)
        a.set_xlabel("Temperature (°C)")
        a.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    spice.save(fig, "m2_retention.png")
    plt.close(fig)
    return table, vfail1


def retention_check(table):
    """Cross-check the leakage integral with a real transient at 125 C (baseline, '1' standby)."""
    tpred = table[(0, 0, 125)]
    nm = "m2_ret_tran"
    body = (".option gmin=1e-18\nVBL bl 0 0.9\nVWL wl 0 0\nVCP vcp 0 0.9\n"
            + spice.fet("ma", "bl", "wl", "sn", "0", "n", 0.42, 0.15)
            + f"CS sn vcp {CS}\n.ic v(sn)=1.8\n"
            + f".control\ntran {tpred/500:.3g} {tpred*1.5:.3g}\n"
            + "meas tran t_cross when v(sn)=VFAIL fall=1\n.endc\n")
    return nm, body, tpred


# ------------------------------------------------------------- (c) corners
def corners(plt, labels, flavor="svt"):
    extra = SA_OPTIONS[flavor]
    jobs, idx = [], []
    for c in spice.CORNERS:
        for t in TEMPS:
            for vdd in VDDS:
                p = dict(extra, vdd=vdd, vpp=round(vdd + 0.8, 3))
                tag = f"{c}_t{t}_v{vdd}".replace("-", "m").replace(".", "p")
                for vin, rn in ((vdd, "r1"), (0.0, "r0")):
                    j = read_job(f"m2_pvt_{flavor}_{tag}_{rn}", vin, p)
                    j.update(corner=c, temp=t)
                    jobs.append(j)
                    idx.append((c, t, vdd, rn))
    res = spice.run_many(jobs)
    keys = ["dv_share", "tsense", "vsn_restored", "tprecharge"]
    rows = []
    with open(f"{spice.RESULTS}/m2_corners_{flavor}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corner", "temp_C", "vdd", "read"] + keys)
        for (c, t, vdd, rn), r in zip(idx, res):
            row = [c, t, vdd, rn] + [r.get(k, float("nan")) for k in keys]
            w.writerow(row)
            rows.append(row)

    def get(c, t, vdd, rn, k):
        for r in rows:
            if r[0] == c and r[1] == t and abs(r[2] - vdd) < 1e-6 and r[3] == rn:
                return r[4 + keys.index(k)]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6.4))
    panels = [("dv_share", "r0", 1e3, "|ΔV| reading a 0 (mV)", True),
              ("tsense", "r0", 1e9, "Sense time, read 0 (ns)", False),
              ("vsn_restored", "r1", 1, "Restored '1' (V) [VPP = VDD + 0.8]", False),
              ("tprecharge", "r1", 1e12, "Precharge time (ps)", False)]
    xs = np.arange(len(spice.CORNERS))
    nfail = 0
    for ax, (k, rn, sc, title, absval) in zip(axs.flat, panels):
        fails = []
        for i, t in enumerate(TEMPS):
            nom = [get(c, t, 1.8, rn, k) for c in spice.CORNERS]
            lo = [get(c, t, 1.62, rn, k) for c in spice.CORNERS]
            hi = [get(c, t, 1.98, rn, k) for c in spice.CORNERS]
            f = (lambda a: abs(a) * sc) if absval else (lambda a: a * sc)
            nom, lo, hi = map(lambda a: np.array([f(v) for v in a]), (nom, lo, hi))
            off = (i - 1) * 0.22
            ax.vlines(xs + off, np.minimum(lo, hi), np.maximum(lo, hi),
                      color=spice.SERIES[i], lw=1.5, alpha=0.7)
            ax.plot(xs + off, nom, "o", ms=6, color=spice.SERIES[i], label=f"{t} °C")
            for j, c in enumerate(spice.CORNERS):
                if np.isnan(lo[j]) or np.isnan(nom[j]) or np.isnan(hi[j]):
                    fails.append(xs[j] + off)
        if fails:
            top = ax.get_ylim()[1]
            ax.plot(fails, [top] * len(fails), "X", ms=9, color=spice.SERIES[7], clip_on=False,
                    label="did not resolve")
            nfail += len(fails)
        ax.set_xticks(xs, [labels[c] for c in spice.CORNERS], fontsize=8)
        ax.set_title(title)
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    fig.legend(h, l, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.97))
    sa = {"svt": "standard-Vt", "svt2x": "standard-Vt 2x wide", "lvt": "low-Vt"}[flavor]
    fig.suptitle(f"PVT corners, {sa} sense amp: dot = VDD 1.8 V, bar = VDD ±10%", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    spice.save(fig, f"m2_corners_{flavor}.png")
    plt.close(fig)

    # worst cases across everything
    worst = {}
    r0 = [r for r in rows if r[3] == "r0"]
    r1 = [r for r in rows if r[3] == "r1"]
    worst["failed cases"] = [r[:4] for r in rows if np.isnan(r[5])]
    rows_ok = [r for r in rows if not np.isnan(r[5])]
    r0 = [r for r in rows_ok if r[3] == "r0"]
    r1 = [r for r in rows_ok if r[3] == "r1"]
    rows = rows_ok
    worst["min |dv| read0"] = min(r0, key=lambda r: abs(r[4]))
    worst["min dv read1"] = min(r1, key=lambda r: r[4])
    worst["max tsense"] = max(rows, key=lambda r: r[5])
    worst["min restored 1"] = min(r1, key=lambda r: r[6])
    worst["max tprecharge"] = max(rows, key=lambda r: r[7])
    return worst


def main():
    plt = spice.style()
    labels = corner_check()
    a = dv_vs_cs(plt)
    print("dV vs Cs:", a)
    ratio = (0.1526 + 0.1319) / 2 / 0.885
    table, vfail = retention(plt, ratio)
    for k, v in enumerate(VARIANTS):
        for ci, c in enumerate(CONDITIONS):
            print(v[0], "|", c[0], {t: f"{table[(k, ci, t)]:.3g}" for t in RET_TEMPS})
    nm, body, tpred = retention_check(table)
    body = body.replace("VFAIL", f"{vfail:.4f}")
    r = spice.run(nm, body, temp=125)
    print(f"retention check @125C: leakage integral {tpred:.3g}s, transient {r.get('t_cross')}")
    with open(f"{spice.RESULTS}/m2_retention_check.txt", "w") as f:
        f.write(f"vfail={vfail:.4f} V\nintegral={tpred:.4g} s\ntransient={r.get('t_cross')} s\n")
    with open(f"{spice.RESULTS}/m2_worst_case.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sense_amp", "metric", "corner", "temp_C", "vdd", "read", "dv_share",
                     "tsense", "vsn_restored", "tprecharge"])
        for flavor in SA_OPTIONS:
            worst = corners(plt, labels, flavor)
            fails = worst.pop("failed cases")
            print(flavor, "failed:", fails)
            w.writerow([flavor, "failed cases", "; ".join("/".join(map(str, x)) for x in fails) or "none"])
            for k, r in worst.items():
                w.writerow([flavor, k] + r)
                print(flavor, k, r)


if __name__ == "__main__":
    main()
