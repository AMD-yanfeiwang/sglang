import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/tmp/canvas_data_sglr/data.json"))
step, ramp = d["tsStepMin"], d["rampMin"]
b = next(x for x in d["byConc"] if x["conc"] == 512)
arms = [("sglr_rr", "round_robin", "#8c8c8c", 1.6), ("sglr_least", "least_in_flight", "#2166ac", 2.2)]
jobs = {a: next(r["job"] for r in d["rows"][a] if r["conc"] == 512) for a, *_ in arms}
n = len(b["ts"]["sglr_rr"]["maxMean"])
xs = [i * step for i in range(n)]
after = [k for k, x in enumerate(xs) if x >= ramp]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
FOOT = "DeepSeek-V4-Pro AgentX, 1P1D, 8 decode DP ranks, MI355X, sgl-router --dp-aware."

# Figure 1: per-rank outstanding / waiting on KV / running, both policies.
rows = [("out", "Outstanding requests"), ("pend", "Waiting on KV transfer"), ("run", "Running requests")]
fig, axes = plt.subplots(3, 2, figsize=(13, 9.6), sharex=True, constrained_layout=True)
cmap = plt.get_cmap("tab10")
for ri, (metric, label) in enumerate(rows):
    top = max(max(max(r) for r in b["ts"][a][metric]) for a, *_ in arms)
    for ci, (arm, name, _, _) in enumerate(arms):
        ax = axes[ri][ci]
        ranks = b["ts"][arm][metric]
        mean = [sum(r[k] for r in ranks) / len(ranks) for k in range(n)]
        span = st.mean(max(r[k] for r in ranks) - min(r[k] for r in ranks) for k in after)
        ax.axvspan(0, ramp, color="#f0f0f0", zorder=0)
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
        for i, r in enumerate(ranks):
            ax.plot(xs, r, color=cmap(i), lw=1.1, label=f"DP{i}")
        ax.plot(xs, mean, color="black", lw=1.8, ls="--", label="8-rank mean")
        ax.set_ylim(0, top * 1.12)
        ax.text(0.99, 0.97, f"after ramp: ranks span {span:.1f} req (max − min), mean {st.mean(mean[k] for k in after):.1f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#333333")
        if ri == 0:
            ax.set_title(name, loc="left", fontsize=12, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(f"{label}\nper rank")
        if ri == 2:
            ax.set_xlabel("Minutes since profiling start")
            ax.set_xticks(range(0, 61, 10))
axes[0][0].text(ramp / 2, axes[0][0].get_ylim()[1] * 0.97, "ramp-up", ha="center", va="top", fontsize=9, color="#777777")
handles, labels = axes[2][1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=9, frameon=False, fontsize=9.5)
fig.suptitle("Per-rank decode load, concurrency 512: outstanding = waiting on KV transfer + running", fontsize=13, x=0.01, ha="left")
fig.text(0.01, -0.075,
         f"{FOOT} 60 s centered rolling mean of per-second counts from the decode log, every minute; prealloc, retracted and "
         f"queue stayed 0.\nSlurm jobs {jobs['sglr_rr']} (round_robin) / {jobs['sglr_least']} (least_in_flight), "
         "AIPerf profiling window (3600 s).",
         fontsize=8.5, color="#555555", ha="left", va="top")
fig.savefig("/tmp/pr_assets_sglr/sglr-c512-per-rank-outstanding-waiting-running.png", dpi=140, bbox_inches="tight")

# Figure 2: both policies over the hour.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, constrained_layout=True)
for ax in (ax1, ax2):
    ax.axvspan(0, ramp, color="#f0f0f0", zorder=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
for arm, name, color, lw in arms:
    ax1.plot(xs, b["ts"][arm]["maxMean"], color=color, lw=lw, label=name)
    ax2.plot(xs, b["ts"][arm]["outCv"], color=color, lw=lw, label=name)
ax1.set_ylabel("Busiest rank / mean\nrunning requests (ratio)")
ax2.set_ylabel("Outstanding-request CV\nacross ranks")
ax1.set_ylim(bottom=1.0)
ax2.set_ylim(bottom=0)
ax2.set_xlabel("Minutes since profiling start")
ax2.set_xticks(range(0, 61, 10))
ax1.text(ramp / 2, ax1.get_ylim()[1] * 0.995, "ramp-up", ha="center", va="top", fontsize=9, color="#777777")
ax1.legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
fig.suptitle("round_robin vs least_in_flight over the hour, concurrency 512", fontsize=13, x=0.01, ha="left")
fig.text(0.01, -0.035,
         f"{FOOT} 60 s centered rolling mean of per-second values, every minute.\nSeparate runs aligned by time since "
         f"profiling start (slurm jobs {jobs['sglr_rr']} / {jobs['sglr_least']}).",
         fontsize=8.5, color="#555555", ha="left", va="top")
fig.savefig("/tmp/pr_assets_sglr/sglr-c512-over-the-hour.png", dpi=150, bbox_inches="tight")
print("done")
