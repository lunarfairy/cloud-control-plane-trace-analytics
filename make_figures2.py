"""Regenerate Figures 1-3 of the manuscript from results/*.json into tex/figures/ (PDF and PNG)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({"axes.labelsize": 8.0, "axes.linewidth": 0.6, "axes.spines.right": False, "axes.spines.top": False,
                     "axes.titlelocation": "left", "axes.titlesize": 8.0, "figure.dpi": 200.0, "font.size": 8.0, "legend.fontsize": 7.0,
                     "legend.frameon": False, "lines.linewidth": 1.2, "patch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.bbox": "tight", "savefig.dpi": 300.0, "xtick.labelsize": 6.0, "xtick.major.size": 3.0, "xtick.major.width": 0.6,
                     "ytick.labelsize": 6.0, "ytick.major.size": 3.0, "ytick.major.width": 0.6})
J = lambda n: json.load(open(os.path.join("results", n), encoding="utf8"))
OUT = os.path.join("tex", "figures")
BLUE, ORANGE = "#1f5fa8", "#c46a00"


def panel_letter(ax, s):
    ax.text(-0.18, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="left")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    boxes = [("1  Collection", ["O1 event history", "     (reads + writes)", "O2 write-only trail", "O3 polls every Δ", "O4 trail + S3 data events"]),
             ("2  State\n    reconstruction", ["replay successful writes", "add implicit resources", "re-sync from snapshots", "→ resource graph"]),
             ("3  Error-state\n    semantics", ["12 declarative invariants", "F1–F7 functional", "S1–S5 silent", "stack located by wiring"]),
             ("4  Episodes and\n    indicators", ["episodes and persistence", "strategy H / T / D / forum", "debugging-phase bins", "trace features (RQ4)"])]
    w, gap, x0 = 22.5, 2.3, 1
    for k, (t, lines) in enumerate(boxes):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 19), w, 23.5, boxstyle="round,pad=0.3,rounding_size=1", fc="#e8eef6", ec=BLUE, lw=1.1))
        ax.text(x + 1, 41.3, t, fontsize=6.8, weight="bold", color=BLUE, va="top", linespacing=1.2)
        for m, line in enumerate(lines):
            ax.text(x + 1, 33.3 - m * 2.75, line, fontsize=5.6, va="top")
        if k < 3:
            ax.annotate("", xy=(x + w + gap - 0.2, 30.5), xytext=(x + w + 0.3, 30.5), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1))
        ax.annotate("", xy=(x + w / 2, 18.6), xytext=(x + w / 2, 15.2), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1))
    ax.add_patch(FancyBboxPatch((x0, 1), 4 * w + 3 * gap, 13.5, boxstyle="round,pad=0.3,rounding_size=1", fc="white", ec=ORANGE, lw=1.1, ls="--"))
    ax.text(x0 + 1, 13.0, "Validation design (this study)", fontsize=6.8, weight="bold", color=ORANGE, va="top")
    ax.text(x0 + 1, 9.2, "2,000 simulated learners (skill, misconceptions, strategy, timetable)\nissue real AWS SDK calls against an emulated control plane,\nrecorded in CloudTrail form.", fontsize=5.6, va="top", linespacing=1.35)
    ax.text(x0 + 50, 9.2, "Ground truth: invariants evaluated on emulator state after every\nwrite, plus hidden labels (phase, strategy, attempt). Each layer\nis scored against it (RQ1–RQ5).", fontsize=5.6, va="top", linespacing=1.35)
    save(fig, "fig1_framework")


def fig2():
    NUM, POLL, LOSS = J("numbers.json"), J("new_part1.json")["poll"], J("loss_all.json")
    D = [1, 5, 15, 60, 240, 1440]
    cF, cS = BLUE, ORANGE
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1, 1.15], "wspace": 0.45})
    ax = axs[0]
    for g, c, lab in (("F", cF, "Functional"), ("S", cS, "Silent")):
        obs = np.array([POLL[str(d)][f"recall_{g}"] for d in D])
        lo = np.array([POLL[str(d)][f"ci_{g}"][0] for d in D]); hi = np.array([POLL[str(d)][f"ci_{g}"][1] for d in D])
        ax.plot(D, [POLL[str(d)][f"pred_{g}"] for d in D], "-", color=c, lw=1.2, alpha=0.8)
        ax.errorbar(D, obs, yerr=[np.clip(obs - lo, 0, None), np.clip(hi - obs, 0, None)], fmt="o", ms=3.5, color=c, capsize=0)
        ax.text(1400, obs[-1] + (0.05 if g == "S" else -0.08), lab, color=c, fontsize=7, ha="right")
    ax.plot(D, [NUM["a3"][str(d)]["episode_count_ratio"] for d in D], "--", color="0.45", lw=1)
    ax.text(1.2, 0.08, "dashed: detected/true count", color="0.35", fontsize=6)
    ax.set_xscale("log"); ax.set_xticks(D); ax.set_xticklabels([str(d) for d in D]); ax.minorticks_off()
    ax.set_ylim(0, 1.05); ax.set_xlabel("Polling interval Δ (min)"); ax.set_ylabel("Episode recall")
    ax.set_title("Snapshots miss what is short", loc="left")
    ax = axs[1]
    bins = ["<5 min", "5-30 min", "0.5-2 h", ">12 h"]
    bc = POLL["60"]["by_cfg"]
    x = np.arange(len(bins))
    ax.bar(x, [bc[b]["recall"] for b in bins], color=cF, alpha=0.75, width=0.6)
    ax.plot(x, [bc[b]["pred"] for b in bins], "D", color="k", ms=3.5)
    for i, b in enumerate(bins):
        ax.text(i, max(bc[b]["recall"], bc[b]["pred"]) + 0.04, f"n={bc[b]['n']}", ha="center", fontsize=6)
    ax.set_xlim(-0.6, len(bins) - 0.4); ax.set_xticks(x); ax.set_xticklabels(["<5\nmin", "5–30\nmin", "0.5–2\nh", ">12\nh"]); ax.set_ylim(0, 1.15)
    ax.set_ylabel("Recall at Δ = 60 min"); ax.set_xlabel("Configuration-violation duration")
    ax.set_title("Uniform-phase prediction", loc="left"); ax.text(0.03, 0.80, "◆ predicted", transform=ax.transAxes, fontsize=6)
    ax = axs[2]
    modes = [("iid", "random"), ("burst", "burst"), ("create_only", "creates"), ("delete_only", "deletes"), ("reorder", "reorder"), ("duplicate", "duplicate")]
    for j, (rs, col, mk, lab) in enumerate((("none", "0.45", "o", "replay only"), ("60", cF, "s", "+ hourly resync"), ("15", "#6fa8dc", "^", "+ 15-min resync"))):
        vals = [LOSS[f"{m}|0.05|{rs}"]["agreement"] for m, _ in modes]
        xx = np.arange(len(modes)) + (j - 1) * 0.22
        ax.errorbar(xx, [v[0] for v in vals], yerr=[[max(v[0] - v[1], 0) for v in vals], [max(v[2] - v[0], 0) for v in vals]],
                    fmt=mk, ms=3.8, color=col, lw=0.8, capsize=0, label=lab)
    ax.set_ylim(0.82, 1.01); ax.set_xlim(-0.5, 5.5); ax.set_xticks(np.arange(len(modes))); ax.set_xticklabels([m for _, m in modes], rotation=35, ha="right")
    ax.set_ylabel("State agreement"); ax.set_title("Periodic resync repairs event loss", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.3)
    for a, letter in zip(axs, "abc"):
        panel_letter(a, letter)
    save(fig, "fig2_channels")


def fig3():
    RQ4 = J("rq4_results.json")
    B, PB = RQ4["explanatory_designB_primary"]["O1"], RQ4["predictive_designB"]["lam0.0"]
    NS, NP = [100, 200, 400, 700], [200, 400, 700]
    designs = [("latent_MML", "Latent MIMIC", BLUE, "o", "-"), ("oracle_glm", "Oracle GLM", "0.35", "s", ":"), ("pretest_IV", "Pre-test IV", ORANGE, "^", "-"),
               ("oracle_linear", "Oracle linear", "0.6", "v", ":"), ("pretest_OLS", "Pre-test OLS", "#8e5ea2", "D", "--"), ("prior_grade_OLS", "Prior-grade OLS", "#b8336a", "P", "--")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.5})
    ax = axs[0]
    for k, lab, c, mk, ls in designs:
        y = np.array([B[k][f"N{n}"]["size"] for n in NS]); e = np.array([B[k][f"N{n}"]["size_mcse"] for n in NS])
        ax.errorbar(NS, y, yerr=1.96 * e, fmt=mk + ls, color=c, ms=3.5, lw=1, capsize=0, label=lab)
    ax.text(400, 1.15, "prior-grade OLS", color="#b8336a", fontsize=6, ha="center"); ax.text(330, 0.5, "pre-test OLS", color="#8e5ea2", fontsize=6, ha="left")
    ax.axhline(0.05, color="k", lw=0.6); ax.text(690, 0.042, "nominal 0.05", fontsize=6, ha="right", va="top")
    ax.set_yscale("log"); ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"]); ax.minorticks_off()
    ax.set_ylim(0.025, 1.6); ax.set_xticks(NS); ax.set_xlabel("Cohort size N"); ax.set_ylabel("Type I error at λ = 0")
    ax.set_title("OLS designs over-reject", loc="left")
    ax = axs[1]
    for k, lab, c, mk, ls in designs[:4]:
        ax.plot(NS, [B[k][f"N{n}"]["power_size_adj_lam0.3"] for n in NS], mk + ls, color=c, ms=3.5, lw=1, label=lab)
    ax.set_ylim(0, 1.05); ax.set_xticks(NS); ax.set_xlabel("Cohort size N"); ax.set_ylabel("Size-adjusted power, λ = 0.3")
    ax.set_title("Latent model outpowers IV", loc="left"); ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axs[2]
    for base, c, mk in (("G", "#b8336a", "P"), ("P", "#8e5ea2", "D"), ("theta_oracle", "0.35", "o")):
        y = np.array([PB[base]["O1"][f"N{n}"]["delta_R2cv_mean"] for n in NP]); e = np.array([PB[base]["O1"][f"N{n}"]["delta_R2cv_mcse"] for n in NP])
        ax.errorbar(NP, y, yerr=1.96 * e, fmt=mk + "-", color=c, ms=3.5, lw=1, capsize=0)
    ax.text(450, 0.168, "over prior grade", color="#b8336a", fontsize=6, ha="center")
    ax.text(450, 0.020, "over pre-test", color="#8e5ea2", fontsize=6, ha="center")
    ax.text(450, -0.013, "over true skill", color="0.35", fontsize=6, ha="center")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.02, 0.18)
    ax.set_xticks(NP); ax.set_xlim(150, 750); ax.set_xlabel("Cohort size N"); ax.set_ylabel("Gain in cross-validated R² at λ = 0")
    ax.set_title("Prediction gain at λ = 0", loc="left")
    for a, letter in zip(axs, "abc"):
        panel_letter(a, letter)
    save(fig, "fig3_designs")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1(); fig2(); fig3()
    print("figures written")
