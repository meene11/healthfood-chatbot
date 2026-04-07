"""실험 11 (쿼리 확장) 비교 차트 생성"""
import json, matplotlib
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
TS = Path(BASE_DIR / "exp11_timestamp.txt").read_text().strip()

MODELS = [
    {"key": "gpt_4o_mini",  "label": "GPT-4o-mini",      "color": "#4472C4"},
    {"key": "groq_llama",   "label": "Llama 3.3 70B",     "color": "#ED7D31"},
    {"key": "gemini_flash", "label": "Gemini Flash Lite", "color": "#A9D18E"},
]

EXP9  = {"gpt_4o_mini":  {"avg_quality":1.048,"hit5":0.095,"mrr":0.048,"avg_time":4.17},
          "groq_llama":   {"avg_quality":0.595,"hit5":0.095,"mrr":0.048,"avg_time":2.63},
          "gemini_flash": {"avg_quality":1.262,"hit5":0.095,"mrr":0.048,"avg_time":7.77}}
EXP10 = {"gpt_4o_mini":  {"avg_quality":1.262,"hit5":0.071,"mrr":0.040,"avg_time":7.57},
          "groq_llama":   {"avg_quality":0.619,"hit5":0.190,"mrr":0.151,"avg_time":5.06},
          "gemini_flash": {"avg_quality":1.381,"hit5":0.167,"mrr":0.106,"avg_time":19.84}}

exp11 = []
for m in MODELS:
    d = json.loads((BASE_DIR / f"exp11_{m['key']}_{TS}.json").read_text(encoding="utf-8"))
    d["color"] = m["color"]; d["key"] = m["key"]
    exp11.append(d)

labels = [s["label"] for s in exp11]
colors = [s["color"] for s in exp11]

# ── Chart 1: 실험 9/10/11 품질 3단계 비교 ───────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("실험 9→10→11 답변 품질 변화 (LLM-as-Judge 0~3점)", fontsize=13, fontweight="bold")

x = np.arange(len(labels)); w = 0.25
data = {
    "실험 9\n(기준)":    [EXP9[m["key"]]["avg_quality"]  for m in MODELS],
    "실험 10\n(HyDE)":   [EXP10[m["key"]]["avg_quality"] for m in MODELS],
    "실험 11\n(쿼리확장)":[s["avg_quality"]               for s in exp11],
}
offsets = [-w, 0, w]
exp_colors = ["#AAAAAA", "#5B9BD5", "#FF7043"]
for (label, vals), offset, ec in zip(data.items(), offsets, exp_colors):
    bars = ax.bar(x + offset, vals, w, label=label, color=ec, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0, 2.0); ax.set_ylabel("평균 품질 (0~3)", fontsize=11)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.4)
plt.tight_layout(); fig.savefig(BASE_DIR / "exp11_chart1_quality_trend.png"); plt.close()
print("저장: exp11_chart1_quality_trend.png")

# ── Chart 2: 쿼리 확장 효과 (실험 9 대비 변화량) ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle("실험 11 쿼리 확장 효과 (실험 9 대비 변화량)", fontsize=13, fontweight="bold")

diffs = {
    "품질 변화": [s["avg_quality"] - EXP9[m["key"]]["avg_quality"] for s, m in zip(exp11, MODELS)],
    "Hit@5 변화": [s["hit5"] - EXP9[m["key"]]["hit5"] for s, m in zip(exp11, MODELS)],
    "응답시간 증가 (초)": [s["avg_time"] - EXP9[m["key"]]["avg_time"] for s, m in zip(exp11, MODELS)],
}
for ax, (title, vals) in zip(axes, diffs.items()):
    bar_colors = ["#2CA02C" if v >= 0 else "#D62728" for v in vals]
    if "시간" in title:
        bar_colors = ["#D62728" for _ in vals]
    bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+(0.01 if v>=0 else -0.02),
                f"{v:+.3f}" if "시간" not in title else f"+{v:.1f}s",
                ha="center", va="bottom" if v>=0 else "top", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=11); ax.set_xticklabels(labels, fontsize=9); ax.grid(axis="y", alpha=0.4)

plt.tight_layout(); fig.savefig(BASE_DIR / "exp11_chart2_improvement.png"); plt.close()
print("저장: exp11_chart2_improvement.png")

# ── Chart 3: 최종 종합 대시보드 (실험 9~11 GPT, Gemini 비교) ───────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("실험 9~11 종합 비교 대시보드", fontsize=14, fontweight="bold")

# 왼쪽위: GPT-4o-mini 품질 트렌드
ax = axes[0][0]
gpt_q = [EXP9["gpt_4o_mini"]["avg_quality"], EXP10["gpt_4o_mini"]["avg_quality"],
         next(s["avg_quality"] for s in exp11 if s["key"]=="gpt_4o_mini")]
gem_q = [EXP9["gemini_flash"]["avg_quality"], EXP10["gemini_flash"]["avg_quality"],
         next(s["avg_quality"] for s in exp11 if s["key"]=="gemini_flash")]
lam_q = [EXP9["groq_llama"]["avg_quality"],  EXP10["groq_llama"]["avg_quality"],
         next(s["avg_quality"] for s in exp11 if s["key"]=="groq_llama")]
exps = ["실험9\n(기준)", "실험10\n(HyDE)", "실험11\n(쿼리확장)"]
ax.plot(exps, gpt_q, "o-", color="#4472C4", label="GPT-4o-mini", linewidth=2, markersize=8)
ax.plot(exps, gem_q, "s-", color="#A9D18E", label="Gemini Flash", linewidth=2, markersize=8)
ax.plot(exps, lam_q, "^-", color="#ED7D31", label="Llama 3.3 70B", linewidth=2, markersize=8)
for vals, c in zip([gpt_q, gem_q, lam_q], ["#4472C4","#A9D18E","#ED7D31"]):
    for i, v in enumerate(vals):
        ax.text(i, v+0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=c)
ax.set_title("답변 품질 트렌드 (0~3)", fontsize=11); ax.set_ylim(0, 2.0)
ax.legend(fontsize=9); ax.grid(alpha=0.4)

# 오른쪽위: Hit@5 트렌드
ax = axes[0][1]
gpt_h = [EXP9["gpt_4o_mini"]["hit5"], EXP10["gpt_4o_mini"]["hit5"],
         next(s["hit5"] for s in exp11 if s["key"]=="gpt_4o_mini")]
gem_h = [EXP9["gemini_flash"]["hit5"], EXP10["gemini_flash"]["hit5"],
         next(s["hit5"] for s in exp11 if s["key"]=="gemini_flash")]
lam_h = [EXP9["groq_llama"]["hit5"],  EXP10["groq_llama"]["hit5"],
         next(s["hit5"] for s in exp11 if s["key"]=="groq_llama")]
ax.plot(exps, gpt_h, "o-", color="#4472C4", linewidth=2, markersize=8)
ax.plot(exps, gem_h, "s-", color="#A9D18E", linewidth=2, markersize=8)
ax.plot(exps, lam_h, "^-", color="#ED7D31", linewidth=2, markersize=8)
ax.set_title("Hit@5 트렌드 (검색 품질)", fontsize=11); ax.set_ylim(0, 0.3)
ax.grid(alpha=0.4)

# 왼쪽아래: 최종 실험11 품질 분포
ax = axes[1][0]
score_colors = {"0":"#D62728","1":"#FF7F0E","2":"#1F77B4","3":"#2CA02C"}
score_labels = {"0":"0점","1":"1점","2":"2점","3":"3점"}
bottoms = np.zeros(len(labels))
for sc in ["0","1","2","3"]:
    vals = [s["quality_dist"].get(sc,0)/s["n_valid"]*100 for s in exp11]
    ax.bar(range(len(labels)), vals, bottom=bottoms, color=score_colors[sc], alpha=0.85, label=score_labels[sc])
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 5:
            ax.text(xi, b+v/2, f"{v:.0f}%", ha="center", va="center", fontsize=9, fontweight="bold", color="white")
    bottoms += np.array(vals)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0,105); ax.set_title("실험 11 품질 분포", fontsize=11)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# 오른쪽아래: 응답 시간 비교
ax = axes[1][1]
t9  = [EXP9[m["key"]]["avg_time"]  for m in MODELS]
t10 = [EXP10[m["key"]]["avg_time"] for m in MODELS]
t11 = [s["avg_time"] for s in exp11]
x2 = np.arange(len(labels)); w2 = 0.25
b1 = ax.bar(x2-w2, t9,  w2, color=colors, alpha=0.35, hatch="\\\\", label="실험9")
b2 = ax.bar(x2,    t10, w2, color=colors, alpha=0.60, hatch="//",   label="실험10(HyDE)")
b3 = ax.bar(x2+w2, t11, w2, color=colors, alpha=0.90,               label="실험11(쿼리확장)")
for b, v in zip(b3, t11):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x2); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("응답 시간 비교 (초)", fontsize=11); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.4)

plt.tight_layout(); fig.savefig(BASE_DIR / "exp11_chart3_dashboard.png"); plt.close()
print("저장: exp11_chart3_dashboard.png")
print("\n모든 차트 생성 완료!")
