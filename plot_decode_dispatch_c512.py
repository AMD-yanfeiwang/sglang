import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/tmp/pr_assets_decgate/data.json"))
step, ramp = d["step"] / 60, d["ramp"] / 60
S = d["series"]
arms = [("base", "--decode-dispatch-limit unset", "#2166ac", 1.6),
        ("dgate", "--decode-dispatch-limit 8", "#d6604d", 2.0)]
n = len(S["base"]["maxMean"])
xs = [i * step for i in range(n)]
after = [k for k, x in enumerate(xs) if x >= ramp]
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
FOOT = ("DeepSeek-V4-Pro AgentX, 1P1D, 8 decode DP ranks, MI355X, sgl-router --dp-aware "
        "(decode ranks by least_in_flight).")

# Figure 1: per-rank outstanding / waiting on KV / running, both arms.
rows = [("out", "Outstanding requests"), ("pend", "Waiting on KV transfer"), ("run", "Running requests")]
fig, axes = plt.subplots(3, 2, figsize=(13, 9.6), sharex=True, constrained_layout=True)
cmap = plt.get_cmap("tab10")
for ri, (metric, label) in enumerate(rows):
    top = max(max(max(r) for r in S[a][metric]) for a, *_ in arms)
    for ci, (arm, name, _, _) in enumerate(arms):
        ax = axes[ri][ci]
        ranks = S[arm][metric]
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
fig.suptitle("Per-rank decode load, concurrency 512: outstanding = waiting on KV transfer + running",
             fontsize=13, x=0.01, ha="left")
fig.text(0.01, -0.075,
         f"{FOOT} 60 s centered rolling mean of per-second counts from the decode log, every minute; prealloc, retracted and "
         f"queue stayed 0.\nWith the limit, requests waiting for a decode slot are at the router, not on decode. Slurm jobs "
         f"{S['base']['job']} (unset) / {S['dgate']['job']} (limit 8), AIPerf profiling window (3600 s).",
         fontsize=8.5, color="#555555", ha="left", va="top")
fig.savefig("/tmp/pr_assets_decgate/sglr-decode-dispatch-c512-per-rank.png", dpi=140, bbox_inches="tight")

# Figure 2: over the hour.
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.6), sharex=True, constrained_layout=True)
for ax in (ax1, ax2, ax3):
    ax.axvspan(0, ramp, color="#f0f0f0", zorder=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
for arm, name, color, lw in arms:
    ax1.plot(xs, S[arm]["maxMean"], color=color, lw=lw, label=name)
    ax2.plot(xs, S[arm]["pendTot"], color=color, lw=lw, label=name)
    ax3.plot(xs, S[arm]["prefillQ"], color=color, lw=lw, label=name)
ax1.set_ylabel("Busiest rank / mean\nrunning requests (ratio)")
ax2.set_ylabel("Waiting on KV transfer\non decode (8 ranks)")
ax3.set_ylabel("Prefill queue\nper rank (mean)")
ax1.set_ylim(bottom=1.0)
ax2.set_ylim(bottom=0)
ax3.set_ylim(bottom=0)
ax3.set_xlabel("Minutes since profiling start")
ax3.set_xticks(range(0, 61, 10))
ax1.text(ramp / 2, ax1.get_ylim()[1] * 0.995, "ramp-up", ha="center", va="top", fontsize=9, color="#777777")
ax1.legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
fig.suptitle("--decode-dispatch-limit 8 vs unset over the hour, concurrency 512", fontsize=13, x=0.01, ha="left")
fig.text(0.01, -0.03,
         f"{FOOT} 60 s centered rolling mean of per-second values, every minute.\nPrefill queue from the prefill log's "
         f"#queue-req. Separate runs aligned by time since profiling start (slurm jobs {S['base']['job']} / "
         f"{S['dgate']['job']}).",
         fontsize=8.5, color="#555555", ha="left", va="top")
fig.savefig("/tmp/pr_assets_decgate/sglr-decode-dispatch-c512-over-the-hour.png", dpi=150, bbox_inches="tight")
print("done")
