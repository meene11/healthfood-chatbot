"""
RAG 챗봇 평가 결과 시각화 스크립트
실험 1~7 + QA 쌍 평가 데이터를 기반으로 그래프 생성
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# 한글 폰트 설정 (Windows)
import matplotlib.font_manager as fm

def set_korean_font():
    font_candidates = [
        "Malgun Gothic", "맑은 고딕",
        "NanumGothic", "NanumBarunGothic",
        "AppleGothic", "Noto Sans CJK KR",
    ]
    available = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available:
            plt.rcParams["font.family"] = font
            break
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 실험 데이터 ──────────────────────────────────────────────
experiments = [
    "Exp1\nBaseline",
    "Exp2\n가중치\n50/50",
    "Exp3\nTOP_K=10",
    "Exp4\nBGE\n리랭커",
    "Exp5\nBGE-M3\n임베딩",
    "Exp6\nBGE-M3\n60/40",
    "Exp7\nBGE-M3\n50/50",
]
exp_labels_short = ["Exp1", "Exp2", "Exp3", "Exp4", "Exp5", "Exp6", "Exp7"]

hit1  = [0.471, 0.471, 0.471, 0.588, 0.765, 0.765, 0.765]
hit3  = [0.647, 0.647, 0.647, 0.647, 0.882, 0.882, 0.882]
hit5  = [0.706, 0.706, 0.706, 0.706, 0.941, 0.941, 0.941]
mrr   = [0.564, 0.564, 0.578, 0.629, 0.828, 0.828, 0.828]
sim   = [0.637, 0.455, 0.653, 0.626, 0.420, 0.360, 0.300]

x = np.arange(len(experiments))
colors = {
    "hit1": "#4C72B0",
    "hit3": "#55A868",
    "hit5": "#C44E52",
    "mrr":  "#8172B2",
    "sim":  "#CCB974",
}

# ── 1. Hit@K 비교 (그룹 바 차트) ─────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
w = 0.25
ax.bar(x - w, hit1, w, label="Hit@1", color=colors["hit1"], alpha=0.85)
ax.bar(x,     hit3, w, label="Hit@3", color=colors["hit3"], alpha=0.85)
ax.bar(x + w, hit5, w, label="Hit@5", color=colors["hit5"], alpha=0.85)

# 임베딩 교체 구분선
ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(3.6, 0.97, "임베딩 교체\n(bge-m3)", fontsize=8, color="gray", va="top")

ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("실험별 Hit@K 비교 (판정 질문 17개)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# 값 레이블 (Hit@5만)
for i, v in enumerate(hit5):
    ax.text(x[i] + w, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart1_hit_at_k.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 2. MRR 비교 ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = [colors["mrr"]] * 4 + ["#E56B6F"] * 3
bars = ax.bar(x, mrr, color=bar_colors, alpha=0.85, width=0.5)

ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(3.6, 0.87, "임베딩 교체", fontsize=8, color="gray")

ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=9)
ax.set_ylim(0, 0.95)
ax.set_ylabel("MRR (Mean Reciprocal Rank)", fontsize=11)
ax.set_title("실험별 MRR 비교", fontsize=13, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for bar, v in zip(bars, mrr):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

legend_patches = [
    mpatches.Patch(color=colors["mrr"], label="e5-small 기반 (Exp1~4)"),
    mpatches.Patch(color="#E56B6F",    label="bge-m3 기반 (Exp5~7)"),
]
ax.legend(handles=legend_patches, fontsize=9)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart2_mrr.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 3. 평균 유사도 점수 ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
bar_colors2 = [colors["sim"]] * 4 + ["#A8D8A8"] * 3
bars = ax.bar(x, sim, color=bar_colors2, alpha=0.85, width=0.5)

ax.axvline(x=3.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
ax.text(3.6, 0.68, "임베딩 교체\n(청크 수 2.1배↑)", fontsize=8, color="gray")

ax.set_xticks(x)
ax.set_xticklabels(experiments, fontsize=9)
ax.set_ylim(0, 0.75)
ax.set_ylabel("평균 Top-1 유사도 점수", fontsize=11)
ax.set_title("실험별 평균 유사도 점수 비교\n(청크 수 증가로 점수 분산 — 낮다고 성능 나쁜 것 아님)", fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for bar, v in zip(bars, sim):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9)

legend_patches2 = [
    mpatches.Patch(color=colors["sim"], label="e5-small / 8,760청크"),
    mpatches.Patch(color="#A8D8A8",    label="bge-m3 / 18,682청크"),
]
ax.legend(handles=legend_patches2, fontsize=9)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart3_avg_similarity.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 4. Exp1 vs Exp4 vs Exp5 핵심 비교 (개선 단계) ────────────
fig, ax = plt.subplots(figsize=(9, 5))
key_exps   = ["Exp1\nBaseline\n(e5-small)", "Exp4\nBGE 리랭커\n(e5-small)", "Exp5\nBGE-M3\n(최종)"]
key_hit1   = [0.471, 0.588, 0.765]
key_hit5   = [0.706, 0.706, 0.941]
key_mrr    = [0.564, 0.629, 0.828]
xk = np.arange(len(key_exps))
wk = 0.22

ax.bar(xk - wk, key_hit1, wk, label="Hit@1", color=colors["hit1"], alpha=0.88)
ax.bar(xk,      key_hit5, wk, label="Hit@5", color=colors["hit5"], alpha=0.88)
ax.bar(xk + wk, key_mrr,  wk, label="MRR",   color=colors["mrr"],  alpha=0.88)

ax.set_xticks(xk)
ax.set_xticklabels(key_exps, fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("핵심 실험 3개 성능 비교 (개선 단계)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# 개선율 화살표 표시
for metric, y_vals, offset in [("Hit@1", key_hit1, -wk), ("MRR", key_mrr, +wk)]:
    for i in range(len(xk) - 1):
        delta = y_vals[i+1] - y_vals[i]
        if delta > 0:
            mid_x = (xk[i] + offset + xk[i+1] + offset) / 2
            ax.annotate(f"+{delta:.3f}", xy=(mid_x, max(y_vals[i], y_vals[i+1]) + 0.04),
                        fontsize=7.5, color="darkgreen", ha="center")

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart4_key_experiments.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 5. 두 평가 방식 비교 (키워드 vs QA 쌍) ───────────────────
fig, ax = plt.subplots(figsize=(8, 5))
metrics  = ["Hit@1", "Hit@3", "Hit@5", "MRR"]
kw_scores = [0.765, 0.882, 0.941, 0.828]   # Exp5 키워드 매칭
qa_scores = [0.024, 0.071, 0.095, 0.048]   # QA 쌍 평가

xm = np.arange(len(metrics))
wm = 0.3
ax.bar(xm - wm/2, kw_scores, wm, label="키워드 매칭 (Exp5, 17문항)", color="#4C72B0", alpha=0.85)
ax.bar(xm + wm/2, qa_scores, wm, label="QA 쌍 chunk_id (42문항)", color="#DD8452", alpha=0.85)

ax.set_xticks(xm)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("두 평가 방식 비교\n(키워드 매칭 vs QA 쌍 chunk_id 일치)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for i, (a, b) in enumerate(zip(kw_scores, qa_scores)):
    ax.text(i - wm/2, a + 0.015, f"{a:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(i + wm/2, b + 0.015, f"{b:.3f}", ha="center", va="bottom", fontsize=8.5)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart5_eval_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 6. QA 쌍 카테고리별 Hit@5 ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
categories = ["다이어트\n논문\n(n=12)", "건강기사\n(n=7)", "기타\n(n=8)", "건강식품\n논문\n(n=10)", "푸드올로지\n(n=5)"]
cat_hit5   = [0.167, 0.143, 0.125, 0.000, 0.000]
cat_colors = ["#55A868" if v > 0 else "#C44E52" for v in cat_hit5]

bars = ax.bar(np.arange(len(categories)), cat_hit5, color=cat_colors, alpha=0.85, width=0.5)
ax.axhline(y=0.095, color="navy", linestyle="--", linewidth=1.3, label=f"전체 평균 Hit@5 = 0.095")
ax.set_xticks(np.arange(len(categories)))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 0.30)
ax.set_ylabel("Hit@5", fontsize=11)
ax.set_title("QA 쌍 평가 — 카테고리별 Hit@5\n(42개 질문, chunk_id 정확 일치 기준)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for bar, v in zip(bars, cat_hit5):
    label = f"{v:.3f}" if v > 0 else "0.000"
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.006, label,
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart6_qa_category.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

# ── 7. 전체 실험 히트맵 스타일 요약 ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))
data = np.array([hit1, hit3, hit5, mrr]).T   # shape (7, 4)
metric_names = ["Hit@1", "Hit@3", "Hit@5", "MRR"]

im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(metric_names)))
ax.set_xticklabels(metric_names, fontsize=11)
ax.set_yticks(np.arange(len(exp_labels_short)))
ax.set_yticklabels(exp_labels_short, fontsize=10)
ax.set_title("실험 1~7 성능 히트맵 (녹색=높음, 빨강=낮음)", fontsize=12, fontweight="bold")

for i in range(len(exp_labels_short)):
    for j in range(len(metric_names)):
        val = data[i, j]
        color = "black" if 0.3 < val < 0.85 else "white"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9.5, color=color)

plt.colorbar(im, ax=ax, shrink=0.8, label="Score")
plt.tight_layout()
out = os.path.join(OUTPUT_DIR, "chart7_heatmap.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out}")

print("\n✓ 모든 차트 생성 완료 (7개)")
