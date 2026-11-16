"""M3: 6T SRAM static characterisation.

  butterfly curves (hold + read) and SNM, write margin, read current
  across process corners and temperature (VDD 1.8 V)

Outputs: results/m3_sram_corners.csv, plots/m3_butterfly.png, plots/m3_sram_corners.png
"""
import csv
import numpy as np
import spice
import sram

TEMPS = [-40, 27, 125]
NOM = [{"a": {}, "b": {}}]


def jobs_for(corner, temp):
    tag = f"{corner}_t{temp}".replace("-", "m")
    return [dict(name=f"m3_bf_{tag}", body=sram.butterfly_body(f"m3_bf_{tag}", NOM),
                 corner=corner, temp=temp),
            dict(name=f"m3_wm_{tag}", body=sram.write_body(f"m3_wm_{tag}", NOM),
                 corner=corner, temp=temp),
            dict(name=f"m3_ir_{tag}", body=sram.read_current_body(), corner=corner, temp=temp)]


def iread_from_log(name):
    for line in open(f"{spice.SIM}/{name}.log"):
        if line.strip().startswith("iread"):
            return float(line.split("=")[1])
    return float("nan")


def analyse(tag):
    x, Y = spice.load(f"m3_bf_{tag}")
    hold, _ = sram.snm(x, Y[:, 0], Y[:, 1])
    read, _ = sram.snm(x, Y[:, 2], Y[:, 3])
    vbl, Q = spice.load(f"m3_wm_{tag}")
    wm = sram.write_margin(vbl, Q[:, 0])
    return hold, read, wm, iread_from_log(f"m3_ir_{tag}")


def main():
    plt = spice.style()
    jobs = []
    for c in spice.CORNERS:
        for t in TEMPS:
            jobs += jobs_for(c, t)
    spice.run_many(jobs)
    rows = []
    for c in spice.CORNERS:
        for t in TEMPS:
            tag = f"{c}_t{t}".replace("-", "m")
            rows.append([c, t, *analyse(tag)])
    with open(f"{spice.RESULTS}/m3_sram_corners.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corner", "temp_C", "hold_snm_V", "read_snm_V", "write_margin_V", "iread_A"])
        w.writerows(rows)
    for r in rows:
        print(r)

    # butterfly plot at tt / 27 C
    x, Y = spice.load("m3_bf_tt_t27")
    hs, (h1, h2) = sram.snm(x, Y[:, 0], Y[:, 1])
    rs, (r1, r2) = sram.snm(x, Y[:, 2], Y[:, 3])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 4.4))
    for ax, ia, ib, s, lab in ((a1, 0, 1, hs, "Hold (WL = 0)"), (a2, 2, 3, rs, "Read (WL = BL = VDD)")):
        ax.plot(x, Y[:, ia], color=spice.SERIES[0], label="Q = f(QB)  (half A)")
        ax.plot(Y[:, ib], x, color=spice.SERIES[1], label="QB = f(Q)  (half B)")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, 1.8)
        ax.set_xlabel("QB (V)")
        ax.set_ylabel("Q (V)")
        ax.set_title(f"{lab}: SNM = {s*1e3:.0f} mV")
        # draw the embedded square in the upper-left eye
        sq = embedded_square(x, Y[:, ia], Y[:, ib], s)
        if sq is not None:
            ax.add_patch(plt.Rectangle(sq, s, s, fill=False, ec=spice.INK2, lw=1, ls="--"))
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("6T SRAM butterfly curves (tt, 27 °C, 1.8 V)")
    fig.tight_layout()
    spice.save(fig, "m3_butterfly.png")
    plt.close(fig)

    # corners plot
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    xs = np.arange(len(spice.CORNERS))
    for ax, k, title in ((axs[0], 3, "Read SNM (mV)"), (axs[1], 2, "Hold SNM (mV)"),
                         (axs[2], 4, "Write margin (mV)")):
        for i, t in enumerate(TEMPS):
            vals = [r[k] * 1e3 for r in rows if r[1] == t]
            ax.plot(xs + (i - 1) * 0.2, vals, "o", ms=6, color=spice.SERIES[i], label=f"{t} °C")
        ax.set_xticks(xs, [spice.corner_labels()[c] for c in spice.CORNERS], fontsize=8)
        ax.set_title(title)
    axs[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("SRAM margins across corners (VDD 1.8 V)")
    fig.tight_layout()
    spice.save(fig, "m3_sram_corners.png")
    plt.close(fig)


def embedded_square(x, ya, yb, s):
    """Lower-left corner of the s x s square that fits best in the upper-left eye (for drawing)."""
    o = np.argsort(yb)
    best, pos = -1e9, None
    for qb0 in np.linspace(0, 1.8 - s, 721):
        top = np.interp(qb0 + s, x, ya)       # curve 1 (Q vs QB) at the square's right edge
        bot = np.interp(qb0, yb[o], x[o])     # curve 2 (Q where QB = qb0) at its left edge
        slack = top - bot - s
        if slack > best:
            best, pos = slack, (qb0, bot)
    return pos


if __name__ == "__main__":
    main()
