"""
실험 9 비교 차트 생성
: 내부 QA 42문항 기반 3개 모델 비교 (검색 + 답변 품질)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path

# ── 폰트 설정 ──────────────────────────────────────────────────────────
def set_korean_font():
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            fm.fontManager.addfont(p)
            prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

BASE_DIR = Path(__file__).resolve().parent
TS = "20260407_141446"

MODELS = [
    {"key": "gpt_4o_mini",   "label": "GPT-4o-mini",        "color": "#4472C4"},
    {"key": "groq_llama",    "label": "Llama 3.3 70B",       "color": "#ED7D31"},
    {"key": "gemini_flash",  "label": "Gemini Flash Lite",   "color": "#A9D18E"},
]

# ── 데이터 로드 ─────────────────────────────────────────────────────────
sums = []
for m in MODELS:
    fp = BASE_DIR / f"exp9_{m['key']}_{TS}.json"
    d  = json.loads(fp.read_text(encoding="utf-8"))
    d["color"] = m["color"]
    sums.append(d)

labels = [s["label"] for s in sums]
colors = [s["color"] for s in sums]

# ── 공통 스타일 ─────────────────────────────────────────────────────────
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150})


# ─────────────────────────────────────────────────────────────────────────
# Chart 1: 검색 품질 (Hit@K + MRR) + 답변 품질
# ─────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("실험 9: 내부 QA 평가 — 검색 & 답변 품질", fontsize=14, fontweight="bold")

# 왼쪽: Hit@K & MRR 막대
ax1 = axes[0]
metrics = ["hit1", "hit3", "hit5", "mrr"]
mlabels = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
x    = np.arange(len(metrics))
w    = 0.25

for i, s in enumerate(sums):
    vals = [s[m] for m in metrics]
    bars = ax1.bar(x + i*w, vals, w, label=s["label"], color=colors[i], alpha=0.85)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax1.set_title("검색 품질 (Retrieval)", fontsize=11)
ax1.set_xticks(x + w)
ax1.set_xticklabels(mlabels)
ax1.set_ylim(0, 0.25)
ax1.set_ylabel("점수")
ax1.legend(fontsize=8)
ax1.grid(axis="y", alpha=0.4)

# 오른쪽: 평균 답변 품질 + 완전정답률
ax2 = axes[1]
qual  = [s["avg_quality"] for s in sums]
perf3 = [s["quality_dist"].get("3", 0) / s["n_valid"] * 100 for s in sums]

x2 = np.arange(len(labels))
w2 = 0.35
b1 = ax2.bar(x2 - w2/2, qual, w2, color=colors, alpha=0.85, label="평균 품질 (0~3)")
b2 = ax2.bar(x2 + w2/2, [p/100*3 for p in perf3], w2,
             color=colors, alpha=0.45, hatch="//", label="완전정답 (3점) 비율×3")

for b, v in zip(b1, qual):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f"{v:.3f}", ha="center", va="bottom", fontsize=9)
for b, v in zip(b2, perf3):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.02,
             f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

ax2.set_title("답변 품질 (LLM-as-Judge)", fontsize=11)
ax2.set_xticks(x2)
ax2.set_xticklabels(labels, fontsize=9)
ax2.set_ylim(0, 3.0)
ax2.set_ylabel("점수")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.4)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp9_chart1_quality.png")
plt.close()
print("저장: exp9_chart1_quality.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 2: 품질 점수 분포 (누적 바)
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("실험 9: 품질 점수 분포 비교", fontsize=13, fontweight="bold")

score_colors = {"0": "#D62728", "1": "#FF7F0E", "2": "#1F77B4", "3": "#2CA02C"}
score_labels = {"0": "0점 (오답)", "1": "1점 (부분)", "2": "2점 (대부분)", "3": "3점 (완전)"}
x = np.arange(len(labels))

bottoms = np.zeros(len(labels))
for sc in ["0", "1", "2", "3"]:
    vals = [s["quality_dist"].get(sc, 0) / s["n_valid"] * 100 for s in sums]
    ax.bar(x, vals, bottom=bottoms, color=score_colors[sc],
           alpha=0.85, label=score_labels[sc])
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v > 4:
            ax.text(xi, b + v/2, f"{v:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
    bottoms += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("비율 (%)")
ax.set_ylim(0, 105)
ax.legend(loc="upper right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp9_chart2_distribution.png")
plt.close()
print("저장: exp9_chart2_distribution.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 3: 카테고리별 답변 품질 히트맵
# ─────────────────────────────────────────────────────────────────────────
cats = sorted(sums[0]["cat_quality"].keys())
cat_labels = [c.replace("_", "\n") for c in cats]
data = np.array([[s["cat_quality"].get(c, 0) for c in cats] for s in sums])

fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle("실험 9: 카테고리별 평균 품질 (0~3)", fontsize=13, fontweight="bold")

im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=3, aspect="auto")
plt.colorbar(im, ax=ax, label="평균 품질")
ax.set_xticks(range(len(cats)))
ax.set_xticklabels(cat_labels, fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)

for i in range(len(labels)):
    for j in range(len(cats)):
        ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                fontsize=11, color="black" if data[i,j] < 2 else "white",
                fontweight="bold")

plt.tight_layout()
fig.savefig(BASE_DIR / "exp9_chart3_heatmap.png")
plt.close()
print("저장: exp9_chart3_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 4: 속도 vs 품질 산점도
# ─────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("실험 9: 응답 시간 vs 답변 품질", fontsize=13, fontweight="bold")

for s in sums:
    ax.scatter(s["avg_time"], s["avg_quality"], s=300, color=s["color"],
               zorder=5, edgecolors="black", linewidths=0.8)
    ax.annotate(s["label"], (s["avg_time"], s["avg_quality"]),
                textcoords="offset points", xytext=(10, 5), fontsize=10)

ax.set_xlabel("평균 응답 시간 (초)", fontsize=11)
ax.set_ylabel("평균 품질 (0~3)", fontsize=11)
ax.set_xlim(0, max(s["avg_time"] for s in sums) * 1.3)
ax.set_ylim(0, 3)
ax.grid(alpha=0.4)
ax.axvline(x=5, color="gray", linestyle="--", alpha=0.5, label="5초 기준선")
ax.legend(fontsize=9)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp9_chart4_speed_quality.png")
plt.close()
print("저장: exp9_chart4_speed_quality.png")


# ─────────────────────────────────────────────────────────────────────────
# Chart 5: 실험 8 vs 실험 9 통합 비교 (답변 품질)
# ─────────────────────────────────────────────────────────────────────────
# Exp 8 결과 (기록된 값)
exp8 = {
    "GPT-4o-mini":      {"avg_quality": 1.581, "hit5": 0.215, "mrr": 0.215},
    "Llama 3.3 70B":    {"avg_quality": 1.629, "hit5": 0.215, "mrr": 0.215},
    "Gemini Flash Lite":{"avg_quality": 1.468, "hit5": 0.215, "mrr": 0.215},
}
exp9 = {s["label"]: {"avg_quality": s["avg_quality"]} for s in sums}

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("실험 8 vs 실험 9 답변 품질 비교", fontsize=13, fontweight="bold")

x   = np.arange(len(labels))
w   = 0.35
b8  = [exp8.get(l, {}).get("avg_quality", 0) for l in labels]
b9  = [exp9.get(l, {}).get("avg_quality", 0) for l in labels]

bars8 = ax.bar(x - w/2, b8, w, color=colors, alpha=0.5, hatch="\\\\", label="실험 8 (조원 QA 62문항)")
bars9 = ax.bar(x + w/2, b9, w, color=colors, alpha=0.9, label="실험 9 (내부 QA 42문항)")

for b, v in zip(bars8, b8):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)
for b, v in zip(bars9, b9):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(0, 3)
ax.set_ylabel("평균 품질 (0~3)", fontsize=11)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
fig.savefig(BASE_DIR / "exp9_chart5_exp8vs9.png")
plt.close()
print("저장: exp9_chart5_exp8vs9.png")

print("\n모든 차트 생성 완료!")
