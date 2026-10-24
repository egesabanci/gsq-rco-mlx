"""Figures for README.md and docs/RESULTS.md, drawn only from the committed run records in results/.

Writes ppl_vs_bpp.png, rco_allocation.png and rco_trace.png next to this file. Every figure has a
white card with a light border, so it reads the same on light and dark GitHub themes.

matplotlib is a documentation dependency only (not in pyproject.toml):
    uv pip install --python .venv/bin/python matplotlib
    .venv/bin/python docs/figures/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parents[1] / "results"
RUNS = RES / "runs"

INK = "#24292f"
MUTED = "#57606a"
BORDER = "#d0d7de"
GPTQ = "#6e7781"
GSQ = "#0969da"
PROXY = "#8250df"
PREFIX = "#bc4c00"
LINEARS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BIT_LEVELS = [2, 3, 4, 5, 6, 8]
BIT_CMAP = ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08519c", "#08306b"]
BIT_TEXT = [INK, INK, INK, "#ffffff", "#ffffff", "#ffffff"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def card(fig) -> None:
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fill=False,
                             edgecolor=BORDER, linewidth=2))


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def ppl_vs_bpp() -> None:
    fp = load(RUNS / "eval_gptqdb_fp.json")["ppl"]
    gptq = {k: load(RUNS / f"eval_gptqdb_{k}.json") for k in ("u2", "u3", "u4", "proxy3", "rco3")}
    gsq_u3 = load(RUNS / "eval_gsq_u3.json")
    gsq_rco3 = load(RUNS / "eval_gsq_rco3.json")
    prefix = load(RUNS / "gsq_rco3_oldrule" / "eval_gsq_rco3.json")
    bpp_u3 = load(RUNS / "gsq_u3" / "run_log.json")["export"]["bits_per_weight"]["native_bpp"]
    bpp_rco3 = load(RUNS / "gsq_rco3" / "run_log.json")["export"]["bits_per_weight"]["native_bpp"]

    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
    fig.patch.set_facecolor("white")
    card(fig)
    style(ax)
    ax.axhline(fp, color=INK, lw=1, ls="--")
    ax.text(2.04, fp * 1.08, f"FP bf16 {fp:.2f}", color=INK, ha="left", fontsize=9)

    uni_x = [gptq["u2"]["bpp_avg_incl_scales"], gptq["u3"]["bpp_avg_incl_scales"],
             gptq["u4"]["bpp_avg_incl_scales"]]
    uni_y = [gptq["u2"]["ppl"], gptq["u3"]["ppl"], gptq["u4"]["ppl"]]
    ax.plot(uni_x, uni_y, color=GPTQ, lw=1.2, zorder=2)
    ax.scatter(uni_x, uni_y, color=GPTQ, s=46, zorder=3,
               label=f"GPTQ uniform 2 / 3 / 4 bit ({gptq['u2']['ppl']:.2f} / {gptq['u3']['ppl']:.2f}"
                     f" / {gptq['u4']['ppl']:.2f})")
    px, py = gptq["proxy3"]["bpp_avg_incl_scales"], gptq["proxy3"]["ppl"]
    ax.scatter([px], [py], color=PROXY, marker="s", s=46, zorder=3,
               label=f"GPTQ, proxy-DP 3.0 bit ({py:.2f})")
    rx, ry = gptq["rco3"]["bpp_avg_incl_scales"], gptq["rco3"]["ppl"]
    ax.scatter([rx], [ry], color=GPTQ, marker="^", s=60, zorder=3,
               label=f"GPTQ, RCO 3.0 allocation ({ry:.2f})")
    ax.scatter([bpp_u3], [gsq_u3["ppl"]], color=GSQ, marker="o", s=46, zorder=4,
               label=f"GSQ uniform 3 bit ({gsq_u3['ppl']:.2f})")
    ax.scatter([bpp_rco3], [gsq_rco3["ppl"]], color=GSQ, marker="D", s=52, zorder=4,
               label=f"GSQ on RCO 3.0 allocation ({gsq_rco3['ppl']:.2f})")
    ax.scatter([bpp_rco3], [prefix["ppl"]], color=PREFIX, marker="x", s=60, zorder=4,
               label=f"GSQ on RCO, before the scale-lr fix ({prefix['ppl']:.2f})")

    ax.set_yscale("log")
    ax.set_xlim(2.0, 4.5)
    ax.set_ylim(10, 2500)
    ax.yaxis.set_major_locator(FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_xlabel("bits per weight incl. fp16 group scales (group size 64)")
    ax.set_ylabel("WikiText-2 test perplexity (log scale)")
    ax.set_title("Qwen2.5-0.5B: perplexity vs bits per weight", fontsize=11, loc="left")
    ax.grid(True, which="major", color=BORDER, lw=0.6)
    ax.legend(loc="upper right", fontsize=8, frameon=True, edgecolor=BORDER, facecolor="white")
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    fig.savefig(HERE / "ppl_vs_bpp.png", facecolor="white")
    plt.close(fig)
    print("ppl_vs_bpp:", {"fp": fp, "gptq": {k: round(v["ppl"], 4) for k, v in gptq.items()},
                          "gsq_u3": gsq_u3["ppl"], "gsq_rco3": gsq_rco3["ppl"], "prefix": prefix["ppl"]})


def rco_allocation() -> None:
    alloc = load(RES / "rco_allocation.json")
    n_layers = 24
    grid = [[alloc["per_linear_bits"][f"{L}.{name}"] for name in LINEARS] for L in range(n_layers)]
    cmap = ListedColormap(BIT_CMAP)
    norm = BoundaryNorm([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 8.5], cmap.N)

    fig, ax = plt.subplots(figsize=(6.4, 8.6), dpi=150)
    fig.patch.set_facecolor("white")
    card(fig)
    style(ax)
    data = np.array(grid, dtype=float)
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for L in range(n_layers):
        for j in range(len(LINEARS)):
            b = grid[L][j]
            col = BIT_TEXT[BIT_LEVELS.index(b)]
            ax.text(j, L, str(b), ha="center", va="center", fontsize=8, color=col)
    ax.set_xticks(range(len(LINEARS)))
    ax.set_xticklabels(LINEARS, rotation=35, ha="right")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([str(L) for L in range(n_layers)])
    ax.set_ylabel("transformer layer")
    ax.set_title(f"RCO allocation, decoded: {alloc['decoded_avg_bits']:.2f} average code bits\n"
                 "(3.0 budget, 168 linears, options 2/3/4/5/6/8)", fontsize=10, loc="left")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=BIT_LEVELS,
                      fraction=0.05, pad=0.03)
    cb.set_label("bits", color=INK)
    cb.ax.tick_params(colors=INK, labelsize=9)
    cb.outline.set_edgecolor(MUTED)
    fig.tight_layout(rect=(0.01, 0.01, 0.99, 0.99))
    fig.savefig(HERE / "rco_allocation.png", facecolor="white")
    plt.close(fig)
    counts = {b: sum(1 for v in alloc["per_linear_bits"].values() if v == b) for b in BIT_LEVELS}
    print("rco_allocation counts:", counts)


def rco_trace() -> None:
    alloc = load(RES / "rco_allocation.json")
    hist = alloc["history"]
    t = [h["t"] for h in hist]
    loss = [h["loss"] for h in hist]
    tau = [h["tau"] for h in hist]
    total_units = alloc["budget"]["total_units"]
    avg_bits = [h["cost"] / total_units for h in hist]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=150, gridspec_kw={"wspace": 0.42})
    fig.patch.set_facecolor("white")
    card(fig)
    for ax in (a1, a2):
        style(ax)
        ax.grid(True, color=BORDER, lw=0.6)

    a1.plot(t, loss, color=GSQ, lw=1.6, label="mean KL over g = 8 sampled allocations")
    a1.set_xlabel("RCO step t")
    a1.set_ylabel("sampled-allocation KL (nats per token)")
    a1.set_title("(a) training objective", fontsize=10, loc="left")
    tw = a1.twinx()
    tw.plot(t, tau, color=MUTED, lw=1.0, ls="--", label="temperature tau")
    tw.set_ylabel("tau (exponential 1.0 -> 0.01)", color=MUTED)
    tw.tick_params(colors=MUTED, labelsize=9)
    tw.spines["top"].set_visible(False)
    tw.set_yscale("log")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=True, edgecolor=BORDER)

    a2.plot(t, avg_bits, color=GSQ, lw=1.6, label="expected cost of p_i (continuous)")
    a2.axhline(alloc["budget"]["avg_bits_target"], color=INK, lw=1, ls="--", label="budget 3.00")
    decoded = alloc["decoded_avg_bits"]
    a2.scatter([t[-1]], [decoded], color=PREFIX, marker="o", s=34, zorder=4,
               label=f"decoded assignment (DP on log p): {decoded:.2f}")
    a2.set_ylim(2.8, 3.2)
    a2.set_xlabel("RCO step t")
    a2.set_ylabel("average bits per weight (code bits)")
    a2.set_title("(b) budget trace", fontsize=10, loc="left")
    a2.legend(loc="lower left", fontsize=8, frameon=True, edgecolor=BORDER)

    fig.suptitle("RCO allocation run: 100 steps, g = 8, lr 0.1, seed 0 (results/rco_allocation.json)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.86, bottom=0.14)
    fig.savefig(HERE / "rco_trace.png", facecolor="white")
    plt.close(fig)
    print("rco_trace:", {"loss_t0": round(loss[0], 4), "loss_last": round(loss[-1], 4),
                         "loss_min": round(min(loss), 4), "avg_bits_min": round(min(avg_bits), 6),
                         "avg_bits_max": round(max(avg_bits), 6), "decoded": round(decoded, 4)})


if __name__ == "__main__":
    ppl_vs_bpp()
    rco_allocation()
    rco_trace()
