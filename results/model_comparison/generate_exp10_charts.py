"""실험 10 (HyDE) 비교 차트 생성"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

def set_korean_font():
    for p in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/gulim.ttc"]:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return

set_korean_font()
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})

BASE_DIR = Path(__file__).resolve().parent
TS = "20260407_144238"

MODELS = [
    {"key": "gpt_4o_mini",  "label": "GPT-4o-mini",       "color": "#4472C4"},
    {"key": "groq_llama",   "label": "Llama 3.3 70B",      "color": "#ED7D31"},
    {"key": "gemini_flash", "label": "Gemini Flash Lite",  "color": "#A9D18E"},
]

# 실험 9 베이스라인 값
EXP9 = {
    "gpt_4o_mini":  {"avg_quality": 1.048, "hit5": 0.095, "mrr": 0.048, "avg_time": 4.17},
    "groq_llama":   {"avg_quality": 0.595, "hit5": 0.095, "mrr": 0.048, "avg_time": 2.63},
    "gemini_flash": {"avg_quality": 1.262, "hit5": 0.095, "mrr": 0.048, "avg_time": 7.77},
}

exp10 = []
for m in MODELS:
    fp = BASE_DIR / f"exp10_{m['key']}_{TS}.json"
    d  = json.loads(fp.read_text(encoding="utf-8"))
    d["color"] = m["color"]
    d["key"]   = m["key"]
    exp10.append(d)

labels = [s["label"] for s in exp10]
colors = [s["color"] for s in exp10]

# ── Chart 1: 실험9 vs 실험10 품질 비교 ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("실험 9 (기준) vs 실험 10 (HyDE) 비교", fontsize=14, fontweight="bold")

x = np.arange(len(labels))
w = 0.35

ax = axes[0]
ax.set_title("답변 품질 (avg quality 0~3)", fontsize=11)
b9  = [EXP9[m["key"]]["avg_quality"] for m in MODELS]
b10 = [s["avg_quality"] for s in exp10]
bars9  = ax.bar(x - w/2, b9,  w, color=colors, alpha=0.45, hatch="\\\\", label="실험 9 (기준)")
bars10 = ax.bar(x + w/2, b10, w, color=colors, alpha=0.9,  label="실험 10 (HyDE)")
for b, v in zip(bars9,  b9):  ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for b, v in zip(bars10, b10): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 3); ax.set_ylabel("점수"); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.4)

ax = axes[1]
ax.set_title("Hit@5 / MRR (검색 품질)", fontsize=11)
b9h  = [EXP9[m["key"]]["hit5"] for m in MODELS]
b10h = [s["hit5"] for s in exp10]
b9m  = [EXP9[m["key"]]["mrr"]  for m in MODELS]
b10m = [s["mrr"]  for s in exp10]
xi = np.arange(len(labels))
ax.bar(xi - 0.3, b9h,  0.2, color=colors, alpha=0.45, hatch="\\\\", label="Hit@5 실험9")
ax.bar(xi - 0.1, b10h, 0.2, color=colors, alpha=0.9,  label="Hit@5 실험10")
ax.bar(xi + 0.1, b9m,  0.2, color=colors, alpha=0.45, hatch="//",   label="MRR 실험9")
ax.bar(xi + 0.3, b10m, 0.2, color=colors, alpha=0.9,  label="MRR 실험10")
for i, (h9, h10, m9, m10) in enumerate(zip(b9h, b10h, b9m, b10m)):
    ax.text(i-0.3, h9+0.005,  f"{h9:.3f}",  ha="center", va="bottom", fontsize=7)
    ax.text(i-0.1, h10+0.005, f"{h10:.3f}", ha="center", va="bottom", fontsize=7)
    ax.text(i+0.1, m9+0.005,  f"{m9:.3f}",  ha="center", va="bottom", fontsize=7)
    ax.text(i+0.3, m10+0.005, f"{m10:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(xi); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 0.35); ax.set_ylabel("점수"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp10_chart1_exp9vs10.png")
plt.close(); print("저장: exp10_chart1_exp9vs10.png")

# ── Chart 2: HyDE 적용 후 변화량 ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("실험 10 HyDE 적용 효과 (실험 9 대비 변화량)", fontsize=13, fontweight="bold")

diffs = {
    "품질 향상": [s["avg_quality"] - EXP9[m["key"]]["avg_quality"] for s, m in zip(exp10, MODELS)],
    "Hit@5 변화": [s["hit5"] - EXP9[m["key"]]["hit5"] for s, m in zip(exp10, MODELS)],
    "MRR 변화":   [s["mrr"]  - EXP9[m["key"]]["mrr"]  for s, m in zip(exp10, MODELS)],
}
for ax, (title, vals) in zip(axes, diffs.items()):
    bar_colors = ["#2CA02C" if v >= 0 else "#D62728" for v in vals]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v + (0.003 if v >= 0 else -0.008),
                f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11); ax.set_xticklabels(labels, fontsize=9); ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp10_chart2_improvement.png")
plt.close(); print("저장: exp10_chart2_improvement.png")

# ── Chart 3: 품질 분포 비교 ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("실험 9 vs 실험 10 품질 분포 비교", fontsize=13, fontweight="bold")

exp9_dist = {"gpt_4o_mini": {"3":8,"2":4,"1":12,"0":18}, "groq_llama": {"3":5,"2":3,"1":4,"0":30}, "gemini_flash": {"3":8,"2":6,"1":17,"0":11}}
score_colors = {"0":"#D62728","1":"#FF7F0E","2":"#1F77B4","3":"#2CA02C"}
score_labels = {"0":"0점(오답)","1":"1점(부분)","2":"2점(대부분)","3":"3점(완전)"}

for ax, (exp_data, title) in zip(axes, [(exp9_dist, "실험 9 (기준)"), (None, "실험 10 (HyDE)")]):
    bottoms = np.zeros(len(labels))
    for sc in ["0","1","2","3"]:
        if exp_data:
            vals = [exp_data[m["key"]].get(sc,0)/42*100 for m in MODELS]
        else:
            vals = [s["quality_dist"].get(sc,0)/s["n_valid"]*100 for s in exp10]
        ax.bar(range(len(labels)), vals, bottom=bottoms, color=score_colors[sc], alpha=0.85, label=score_labels[sc])
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 5:
                ax.text(xi, b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        bottoms += np.array(vals)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105); ax.set_title(title, fontsize=11); ax.set_ylabel("비율 (%)"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp10_chart3_distribution.png")
plt.close(); print("저장: exp10_chart3_distribution.png")

# ── Chart 4: 응답 시간 비교 ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("실험 9 vs 실험 10 응답 시간 비교 (HyDE 오버헤드)", fontsize=12, fontweight="bold")
x = np.arange(len(labels)); w = 0.35
t9  = [EXP9[m["key"]]["avg_time"] for m in MODELS]
t10 = [s["avg_time"] for s in exp10]
b9  = ax.bar(x - w/2, t9,  w, color=colors, alpha=0.45, hatch="\\\\", label="실험 9 (기준)")
b10 = ax.bar(x + w/2, t10, w, color=colors, alpha=0.9,  label="실험 10 (HyDE)")
for b, v in zip(b9,  t9):  ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
for b, v in zip(b10, t10): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10); ax.set_ylabel("평균 응답 시간 (초)"); ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); fig.savefig(BASE_DIR / "exp10_chart4_time.png"); plt.close(); print("저장: exp10_chart4_time.png")

print("\n모든 차트 생성 완료!")
