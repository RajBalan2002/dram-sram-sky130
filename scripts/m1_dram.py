"""M1: DRAM read / restore / precharge / write, plus VPP sweep.

Outputs
  results/m1_read.csv, results/m1_write.csv, results/m1_vpp_sweep.csv
  plots/m1_read1.png, m1_read0.png, m1_precharge_fix.png, m1_write.png, m1_vpp_sweep.png
"""
import csv
import numpy as np
import spice
from dram import column, read_timing, read_measures

SAVE = "v(bl) v(blb) v(sn) v(wl) v(san) v(sap) v(peq) v(csl)"
NAMES = ["BL", "BLB", "SN", "WL", "SAN", "SAP", "PEQ", "CSL"]


def control(meas, tstop, name):
    return (".control\n"
            f"tran 5p {tstop}n\n"
            "let vdiff = v(bl) - v(blb)\nlet avdiff = abs(vdiff)\n"
            f"{meas}"
            f"wrdata {name}.dat {SAVE}\n"
            ".endc\n")


def read_job(name, vinit, p=None):
    p = p or {}
    c = read_timing(p)
    body = column(p, [c]) + f".ic v(sn)={vinit} v(bl)={p.get('vdd', 1.8)/2} v(blb)={p.get('vdd', 1.8)/2}\n"
    meas = read_measures(c, vdd=p.get("vdd", 1.8)) + "meas tran dv_residual find avdiff at=13.9n\n"
    body += control(meas, 14, name)
    return dict(name=name, body=body)


def write_job(name, vinit, data):
    c0, c1 = read_timing({}), read_timing({}, t0=12.0)
    w = dict(data=data, t_on=6.5, t_off=8.5)
    body = column({}, [c0, c1], write=w) + f".ic v(sn)={vinit} v(bl)=0.9 v(blb)=0.9\n"
    meas = (f"meas tran vsn_written find v(sn) at={c0['wl_off'] - 0.1}n\n"
            f"meas tran dv_readback find vdiff at={c1['san'] - 0.1}n\n"
            f"meas tran vbl_final find v(bl) at={c1['wl_off'] - 0.1}n\n"
            f"meas tran twrite trig at=6.5n targ avdiff val=1.44 rise=1 td=6.6n\n")
    body += control(meas, 26, name)
    return dict(name=name, body=body)


def plot_wave(plt, name, title, tmax=14):
    t, V = spice.load(name)
    t = t * 1e9
    fig, (a1, a2) = plt.subplots(2, 1, sharex=True, figsize=(8.5, 5.6),
                                 gridspec_kw=dict(height_ratios=[1, 1.4]))
    for i in (6, 3, 4, 5, 7):
        if np.ptp(V[:, i]) > 1e-3:
            a1.plot(t, V[:, i], label=NAMES[i], lw=1.5)
    a1.set_ylabel("Control (V)")
    a1.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    for i in (0, 1, 2):
        a2.plot(t, V[:, i], label=NAMES[i])
    a2.set_ylabel("Data (V)")
    a2.set_xlabel("Time (ns)")
    a2.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    a2.set_xlim(0, tmax)
    fig.suptitle(title, y=1.06)
    spice.save(fig, f"{name}.png")
    plt.close(fig)


def main():
    plt = spice.style()

    # ---- reads: fixed precharge (PEQ at VPP) and the original weak precharge (PEQ at VDD)
    jobs = [read_job("m1_read1", 1.8), read_job("m1_read0", 0.0),
            read_job("m1_read1_peqvdd", 1.8, {"peq_high": "vdd"}),
            read_job("m1_read0_peqvdd", 0.0, {"peq_high": "vdd"})]
    res = spice.run_many(jobs)
    keys = ["vsn_before", "dv_share", "tsense", "vsn_restored", "vsn_hold", "tprecharge", "dv_residual"]
    with open(f"{spice.RESULTS}/m1_read.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + keys)
        for j, r in zip(jobs, res):
            w.writerow([j["name"]] + [r.get(k, float("nan")) for k in keys])
            print(j["name"], {k: r.get(k) for k in keys})
    plot_wave(plt, "m1_read1", "Read a stored 1 (tt, 27 °C, VPP 2.6 V)")
    plot_wave(plt, "m1_read0", "Read a stored 0 (tt, 27 °C, VPP 2.6 V)")

    # precharge comparison
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for nm, lab, col in (("m1_read1_peqvdd", "PEQ at VDD (1.8 V)", spice.SERIES[1]),
                         ("m1_read1", "PEQ at VPP (2.6 V)", spice.SERIES[0])):
        t, V = spice.load(nm)
        ax.plot(t * 1e9, V[:, 0], color=col, label=f"BL, {lab}")
        ax.plot(t * 1e9, V[:, 1], color=col, ls="--", label=f"BLB, {lab}")
    ax.set_xlim(9.5, 14)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Bitline (V)")
    ax.set_title("Precharge after a read: equalize drive decides tRP")
    ax.legend(loc="upper right", fontsize=8)
    spice.save(fig, "m1_precharge_fix.png")
    plt.close(fig)

    # ---- writes: overwrite a 1 with 0, and a 0 with 1, then read back
    wjobs = [write_job("m1_write0", 1.8, 0), write_job("m1_write1", 0.0, 1)]
    wres = spice.run_many(wjobs)
    wkeys = ["vsn_written", "twrite", "dv_readback", "vbl_final"]
    with open(f"{spice.RESULTS}/m1_write.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case"] + wkeys)
        for j, r in zip(wjobs, wres):
            w.writerow([j["name"]] + [r.get(k, float("nan")) for k in wkeys])
            print(j["name"], {k: r.get(k) for k in wkeys})
    plot_wave(plt, "m1_write0", "Write 0 over a stored 1, then read it back", tmax=24)

    # ---- VPP sweep
    vpps = np.round(np.arange(1.8, 3.21, 0.1), 2)
    vj = [read_job(f"m1_vpp_{v:.1f}", 1.8, {"vpp": float(v)}) for v in vpps]
    vr = spice.run_many(vj)
    # signal the NEXT read would see from the restored level (charge sharing ratio from this run)
    ratio = res[0]["dv_share"] / (1.8 - 0.9)
    rows = []
    for v, r in zip(vpps, vr):
        nxt = (r["vsn_restored"] - 0.9) * ratio
        rows.append([v, r["dv_share"], r["vsn_restored"], nxt])
    with open(f"{spice.RESULTS}/m1_vpp_sweep.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vpp", "dv_share", "vsn_restored", "dv_next_read_est"])
        w.writerows(rows)
    rows = np.array(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.plot(rows[:, 0], rows[:, 2], marker="o", ms=4)
    a1.axhline(1.8, color=spice.INK2, lw=0.8, ls=":")
    a1.set_xlabel("Wordline high level VPP (V)")
    a1.set_ylabel("Restored '1' level (V)")
    a1.set_title("Restore needs VPP ≳ VDD + Vt")
    a2.plot(rows[:, 0], rows[:, 1] * 1e3, marker="o", ms=4, label="this read (cell started at 1.8 V)")
    ok = rows[:, 0] >= 2.1   # below this the cell is barely connected, the estimate is meaningless
    a2.plot(rows[ok, 0], rows[ok, 3] * 1e3, marker="s", ms=4, label="next read (from restored level)")
    for a in (a1, a2):
        a.axvspan(1.75, 2.05, color=spice.GRID, alpha=0.6, lw=0)
    a1.annotate("cell barely\nconnected", (1.9, 1.55), fontsize=8, color=spice.INK2, ha="center")
    a2.set_xlabel("Wordline high level VPP (V)")
    a2.set_ylabel("Bitline signal ΔV (mV)")
    a2.set_title("Read signal vs VPP")
    a2.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    spice.save(fig, "m1_vpp_sweep.png")
    plt.close(fig)
    print(rows)


if __name__ == "__main__":
    main()
