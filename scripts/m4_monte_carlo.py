"""M4: Monte Carlo mismatch.

Mismatch model: every transistor gets an independent threshold shift
  dVt ~ N(0, A_Vt / sqrt(W*L)),  A_Vt from the SKY130 mismatch files.

  (a) 6T SRAM: hold SNM, read SNM, write margin distributions (N cells)
  (b) DRAM sense amp: input-referred offset distribution, standard-Vt and low-Vt
      versions, found per instance by binary search on the bitline split.

Outputs: results/m4_sram_mc.csv, results/m4_sa_offset.csv, results/m4_summary.csv,
         plots/m4_sram_mc.png, plots/m4_sa_offset.png
"""
import csv
import numpy as np
import spice
import sram

N_SRAM = 1000
N_SA = 400
CHUNK = 100
SEED = 130


def sram_cells(rng, n):
    cells = []
    for _ in range(n):
        c = {}
        for side in ("a", "b"):
            c[side] = {name: rng.normal(0, spice.sigma_vt(kind, *sram.SIZES[name]))
                       for name, kind in sram.TX}
        cells.append(c)
    return cells


def sram_mc(plt, rng):
    cells = sram_cells(rng, N_SRAM)
    jobs = []
    for k in range(0, N_SRAM, CHUNK):
        part = cells[k:k + CHUNK]
        jobs.append(dict(name=f"m4_bf_{k}", body=sram.butterfly_body(f"m4_bf_{k}", part, step=0.01)))
        jobs.append(dict(name=f"m4_wm_{k}", body=sram.write_body_tran(f"m4_wm_{k}", part)))
    res = spice.run_many(jobs)
    rows = []
    for j, k in enumerate(range(0, N_SRAM, CHUNK)):
        x, Y = spice.load(f"m4_bf_{k}")
        wres = res[2 * j + 1]
        for i in range(Y.shape[1] // 4):
            h, _ = sram.snm(x, Y[:, 4 * i], Y[:, 4 * i + 1])
            r, _ = sram.snm(x, Y[:, 4 * i + 2], Y[:, 4 * i + 3])
            wm = sram.wm_from_time(wres.get(f"tf{i}", float("nan")))
            rows.append([k + i, h, r, wm])
    rows = np.array(rows)
    with open(f"{spice.RESULTS}/m4_sram_mc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "hold_snm_V", "read_snm_V", "write_margin_V"])
        w.writerows(rows)

    stats = {}
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, col, title in ((axs[0], 2, "Read SNM"), (axs[1], 1, "Hold SNM"), (axs[2], 3, "Write margin")):
        v = rows[:, col] * 1e3
        mu, sd = v.mean(), v.std(ddof=1)
        stats[title] = (mu, sd)
        ax.hist(v, bins=40, color=spice.SERIES[0], edgecolor="white", linewidth=0.5)
        ax.axvline(mu, color=spice.INK, lw=1)
        ax.axvline(mu - 6 * sd, color=spice.SERIES[7], lw=1.2, ls="--")
        ax.set_title(f"{title}: μ {mu:.0f} mV, σ {sd:.1f} mV\nμ−6σ = {mu - 6*sd:.0f} mV (dashed)")
        ax.set_xlabel("mV")
        lo = min(v.min(), mu - 6.5 * sd)
        ax.set_xlim(lo - 10, v.max() + 10)
    axs[0].set_ylabel(f"cells (N = {N_SRAM})")
    fig.suptitle("6T SRAM Monte Carlo, Vt mismatch only (tt, 27 °C, 1.8 V)")
    fig.tight_layout()
    spice.save(fig, "m4_sram_mc.png")
    plt.close(fig)
    return stats


# ------------------------------------------------------------- SA offset
SA = dict(svt=dict(vn="svt", vp="svt", wn=1.0, wp=2.0, lp=0.15),
          svt2x=dict(vn="svt", vp="svt", wn=2.0, wp=4.0, lp=0.15),
          lvt=dict(vn="lvt", vp="lvt", wn=1.0, wp=4.0, lp=0.35))
SA_LABEL = dict(svt="standard-Vt (W 1/2 µm)", svt2x="standard-Vt 2x wide (W 2/4 µm)",
                lvt="low-Vt (W 1/4 µm)")


def sa_body(name, dvts, dvs, flav):
    """N independent latch sense amps; instance i starts with BL-BLB = dvs[i]."""
    f = SA[flav]
    b = ["VDD vdd 0 1.8\n",
         "VSAN san 0 PWL(0 0.9 0.2n 0.9 0.7n 0)\n",
         "VSAP sap 0 PWL(0 0.9 0.7n 0.9 1.2n 1.8)\n"]
    ic, meas = [], []
    for i, (d, dv) in enumerate(zip(dvts, dvs)):
        bl, blb = f"bl{i}", f"blb{i}"
        b.append(f"C{bl} {bl} 0 100f\nC{blb} {blb} 0 100f\n")
        b.append(spice.fet(f"n1_{i}", bl, blb, "san", "0", "n", f["wn"], 0.15, d[0], f["vn"]))
        b.append(spice.fet(f"n2_{i}", blb, bl, "san", "0", "n", f["wn"], 0.15, d[1], f["vn"]))
        b.append(spice.fet(f"p1_{i}", bl, blb, "sap", "vdd", "p", f["wp"], f["lp"], d[2], f["vp"]))
        b.append(spice.fet(f"p2_{i}", blb, bl, "sap", "vdd", "p", f["wp"], f["lp"], d[3], f["vp"]))
        ic.append(f"v({bl})={0.9 + dv / 2:.6f} v({blb})={0.9 - dv / 2:.6f}")
        meas.append(f"let d{i} = v({bl}) - v({blb})\nmeas tran o{i} find d{i} at=3.9n")
    b.append(".ic " + " ".join(ic) + "\n")
    b.append(".control\ntran 10p 4n\n" + "\n".join(meas) + "\n.endc\n")
    return "".join(b)


def sa_offsets(flav, rng, n=N_SA, iters=12, span=0.12):
    f = SA[flav]
    sig = [spice.sigma_vt("n", f["wn"], 0.15, f["vn"])] * 2 + \
          [spice.sigma_vt("p", f["wp"], f["lp"], f["vp"])] * 2
    dvts = rng.normal(0, 1, (n, 4)) * np.array(sig)
    lo, hi = np.full(n, -span), np.full(n, span)
    for it in range(iters):
        mid = (lo + hi) / 2
        jobs = []
        for k in range(0, n, CHUNK):
            jobs.append(dict(name=f"m4_sa_{flav}_{k}",
                             body=sa_body(f"m4_sa_{flav}_{k}", dvts[k:k + CHUNK], mid[k:k + CHUNK], flav)))
        res = spice.run_many(jobs)
        out = np.concatenate([[r[f"o{i}"] for i in range(len(dvts[k:k + CHUNK]))]
                              for k, r in zip(range(0, n, CHUNK), res)])
        ones = out > 0          # latch resolved to BL high
        hi = np.where(ones, mid, hi)
        lo = np.where(ones, lo, mid)
    return (lo + hi) / 2, dvts


def sa_mc(plt, rng):
    data, stats = {}, {}
    for flav in SA:
        off, _ = sa_offsets(flav, rng)
        data[flav] = off
        stats[flav] = (off.mean(), off.std(ddof=1))
        print(flav, "offset mean %.2f mV sigma %.2f mV" % (off.mean() * 1e3, off.std(ddof=1) * 1e3))
    with open(f"{spice.RESULTS}/m4_sa_offset.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance"] + [f"offset_{k}_V" for k in SA])
        for i in range(N_SA):
            w.writerow([i] + [data[k][i] for k in SA])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(-60, 60, 31)
    for k, flav in enumerate(SA):
        lab = SA_LABEL[flav]
        mu, sd = stats[flav]
        ax.hist(data[flav] * 1e3, bins=bins, histtype="step", lw=2, color=spice.SERIES[k],
                label=f"{lab}: σ = {sd*1e3:.1f} mV, 6σ = {6*sd*1e3:.0f} mV")
    ax.set_xlabel("Input-referred offset (mV)  [ΔV needed to flip the decision]")
    ax.set_ylabel(f"instances (N = {N_SA})")
    ax.set_title("Sense-amp offset Monte Carlo (tt, 27 °C, Cbl 100 fF)")
    ax.legend(loc="upper left", fontsize=8)
    spice.save(fig, "m4_sa_offset.png")
    plt.close(fig)
    return stats


def main():
    plt = spice.style()
    rng = np.random.default_rng(SEED)
    s1 = sram_mc(plt, rng)
    s2 = sa_mc(plt, rng)
    with open(f"{spice.RESULTS}/m4_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "mean_mV", "sigma_mV", "mean_minus_6sigma_mV"])
        for k, (mu, sd) in s1.items():
            w.writerow([f"SRAM {k}", round(mu, 1), round(sd, 2), round(mu - 6 * sd, 1)])
        for k, (mu, sd) in s2.items():
            w.writerow([f"SA offset ({k})", round(mu * 1e3, 2), round(sd * 1e3, 2), round(6 * sd * 1e3, 1)])
    print(s1, s2)


if __name__ == "__main__":
    main()
