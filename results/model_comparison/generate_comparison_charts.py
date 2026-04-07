"""
모델 비교 평가 결과 시각화 스크립트
현재 평가 완료: GPT-4o-mini
향후 추가 모델 결과가 생기면 자동으로 비교 차트에 반영됨
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from pathlib import Path
import matplotlib.font_manager as fm

# 한글 폰트
def set_korean_font():
    for font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic"]:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()

OUTPUT_DIR = Path(__file__).parent

# ── 데이터 로드 ───────────────────────────────────────────────────────
def load_all_results() -> list[dict]:
    results = []
    for f in sorted(OUTPUT_DIR.glob("eval_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            results.append(d)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")
    return results

all_results = load_all_results()
print(f"로드된 모델 결과: {len(all_results)}개")
for r in all_results:
    print(f"  - {r['label']}: quality={r['avg_quality']}, coverage={r['avg_coverage']}")

# ── 현재 GPT-4o-mini 상세 데이터 ────────────────────────────────────
current = all_results[0]  # gpt-4o-mini
LABEL   = current["label"]
N       = current["n_valid"]

# 질문별 raw 데이터
items = current["results"]
coverages = [r["coverage"] for r in items if r["quality_score"] >= 0]
qualities = [r["quality_score"] for r in items if r["quality_score"] >= 0]
times     = [r["total_sec"]    for r in items if r["quality_score"] >= 0]
cats      = [r["category"]     for r in items if r["quality_score"] >= 0]

q_dist = current["quality_dist"]
cat_avg = current["cat_avg"]

# ── 차트 1: 답변 품질 분포 (파이 차트) ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

labels_pie = ["3점\n완전정답", "2점\n대부분정확", "1점\n부분정답", "0점\n오답/모름"]
sizes      = [int(q_dist.get("3", 0)), int(q_dist.get("2", 0)),
              int(q_dist.get("1", 0)), int(q_dist.get("0", 0))]
colors_pie = ["#4CAF50", "#8BC34A", "#FFC107", "#F44336"]
explode    = (0.05, 0, 0, 0)

wedges, texts, autotexts = axes[0].pie(
    sizes, labels=labels_pie, colors=colors_pie, explode=explode,
    autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
)
axes[0].set_title(f"답변 품질 분포 (LLM-as-Judge)\n{LABEL} | n={N}", fontsize=12, fontweight="bold")

# 바 차트 (오른쪽)
x = np.arange(4)
bar_labels = ["3점\n(완전정답)", "2점\n(대부분)", "1점\n(부분)", "0점\n(오답)"]
bars = axes[1].bar(x, sizes, color=colors_pie, alpha=0.85, width=0.5)
axes[1].set_xticks(x)
axes[1].set_xticklabels(bar_labels, fontsize=10)
axes[1].set_ylabel("문항 수", fontsize=11)
axes[1].set_title("품질 점수별 문항 수", fontsize=12, fontweight="bold")
axes[1].yaxis.grid(True, alpha=0.3)
axes[1].set_axisbelow(True)
for bar, v in zip(bars, sizes):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.3, str(v),
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
out = OUTPUT_DIR / "chart_qa1_quality_dist.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

# ── 차트 2: 카테고리별 품질 & 커버리지 ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
cat_order = sorted(cat_avg.keys())
cat_quality  = [cat_avg[c] for c in cat_order]

from collections import defaultdict
cat_cov_map = defaultdict(list)
for r in items:
    if r["quality_score"] >= 0:
        cat_cov_map[r["category"]].append(r["coverage"])
cat_cov = [sum(cat_cov_map[c])/len(cat_cov_map[c]) if cat_cov_map[c] else 0 for c in cat_order]
cat_n   = [len(cat_cov_map[c]) for c in cat_order]

xc = np.arange(len(cat_order))
w  = 0.3
b1 = ax.bar(xc - w/2, cat_quality, w, label="평균 품질 (0~3)", color="#4C72B0", alpha=0.85)
b2 = ax.bar(xc + w/2, cat_cov,     w, label="평균 커버리지 (0~1)", color="#55A868", alpha=0.85)

ax.set_xticks(xc)
cat_labels_with_n = [f"{c}\n(n={cat_n[i]})" for i, c in enumerate(cat_order)]
ax.set_xticklabels(cat_labels_with_n, fontsize=10)
ax.set_ylim(0, 3.2)
ax.set_ylabel("점수", fontsize=11)
ax.set_title(f"카테고리별 평균 품질 vs 검색 커버리지\n{LABEL}", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

for bar, v in zip(b1, cat_quality):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9)
for bar, v in zip(b2, cat_cov):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05, f"{v:.2f}",
            ha="center", va="bottom", fontsize=9)

plt.tight_layout()
out = OUTPUT_DIR / "chart_qa2_category.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

# ── 차트 3: 품질 vs 커버리지 산점도 ─────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
jitter_y = np.array(qualities) + np.random.uniform(-0.08, 0.08, len(qualities))

cat_color_map = {"general": "#4C72B0", "diet": "#55A868", "research": "#C44E52", "review": "#8172B2"}
cat_list = list(cat_color_map.keys())
for cat in cat_list:
    idx = [i for i, c in enumerate(cats) if c == cat]
    if idx:
        xp = [coverages[i] for i in idx]
        yp = [jitter_y[i] for i in idx]
        ax.scatter(xp, yp, label=cat, color=cat_color_map[cat], alpha=0.7, s=70)

ax.set_xlabel("검색 커버리지 (source_content 키워드 포함률)", fontsize=11)
ax.set_ylabel("답변 품질 점수 (0~3)", fontsize=11)
ax.set_ylim(-0.3, 3.3)
ax.set_xlim(-0.02, 1.0)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["0 (오답)", "1 (부분)", "2 (대부분)", "3 (완전정답)"])
ax.set_title(f"검색 커버리지 vs 답변 품질 산점도\n{LABEL}", fontsize=12, fontweight="bold")
ax.legend(title="카테고리", fontsize=9)
ax.xaxis.grid(True, alpha=0.3)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# 평균선
ax.axvline(x=np.mean(coverages), color="navy", linestyle="--", alpha=0.5,
           label=f"평균 커버리지 {np.mean(coverages):.2f}")
ax.axhline(y=np.mean(qualities), color="darkred", linestyle="--", alpha=0.5,
           label=f"평균 품질 {np.mean(qualities):.2f}")

plt.tight_layout()
out = OUTPUT_DIR / "chart_qa3_scatter.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

# ── 차트 4: 응답 시간 분포 ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(times, bins=15, color="#4C72B0", alpha=0.8, edgecolor="white")
ax.axvline(x=np.mean(times), color="red", linestyle="--", linewidth=1.5,
           label=f"평균 {np.mean(times):.1f}초")
ax.set_xlabel("응답 시간 (초)", fontsize=11)
ax.set_ylabel("문항 수", fontsize=11)
ax.set_title(f"응답 시간 분포\n{LABEL}", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
out = OUTPUT_DIR / "chart_qa4_response_time.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

# ── 차트 5: 종합 대시보드 (현재 + 모델 비교 계획) ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 왼쪽: 현재 지표 요약 바
metric_names = ["검색\n커버리지", "평균\n품질\n(÷3)", "완전정답\n비율"]
pct_perfect  = int(q_dist.get("3", 0)) / N
metric_vals  = [current["avg_coverage"], current["avg_quality"] / 3.0, pct_perfect]
metric_colors= ["#55A868", "#4C72B0", "#C44E52"]

bars = axes[0].bar(np.arange(3), metric_vals, color=metric_colors, alpha=0.85, width=0.45)
axes[0].set_xticks(np.arange(3))
axes[0].set_xticklabels(metric_names, fontsize=10)
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel("정규화 점수 (0~1)", fontsize=11)
axes[0].set_title(f"현재 모델 핵심 지표\n{LABEL}", fontsize=11, fontweight="bold")
axes[0].yaxis.grid(True, alpha=0.3)
axes[0].set_axisbelow(True)
for bar, v in zip(bars, metric_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.015, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=10)

# 오른쪽: 비교 예정 모델 계획 표
axes[1].axis("off")
plan_data = [
    ["모델", "제공사", "비용\n($/1M 입력)", "특징", "상태"],
    ["GPT-4o-mini", "OpenAI", "$0.15", "현재 사용\n빠름·저렴", "완료"],
    ["GPT-4o", "OpenAI", "$2.50", "최고 성능\n추론력 강함", "예정"],
    ["Claude\nHaiku 4.5", "Anthropic", "$0.08", "가장 빠름\n저렴·한국어 강함", "예정"],
    ["Llama 3.3\n70B (Groq)", "Meta/Groq", "무료", "오픈소스\n무료 API", "예정"],
]
col_widths = [0.2, 0.15, 0.15, 0.28, 0.12]
table = axes[1].table(
    cellText=plan_data[1:],
    colLabels=plan_data[0],
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.3, 1.8)

# 완료된 행 강조
for j in range(len(plan_data[0])):
    table[(1, j)].set_facecolor("#C8E6C9")   # 연초록 (완료)
    for i in range(2, len(plan_data)):
        table[(i, j)].set_facecolor("#FFF9C4")   # 연노랑 (예정)

axes[1].set_title("모델 비교 계획", fontsize=12, fontweight="bold", pad=20)

plt.tight_layout()
out = OUTPUT_DIR / "chart_qa5_dashboard.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

# ── 차트 6: 질문별 품질 히트맵 (카테고리별 정렬) ─────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# 카테고리별로 정렬
sorted_items = sorted(
    [r for r in items if r["quality_score"] >= 0],
    key=lambda x: (x["category"], x["id"])
)
q_scores   = [r["quality_score"] for r in sorted_items]
q_labels   = [f"Q{r['id']}" for r in sorted_items]
q_cats     = [r["category"] for r in sorted_items]

# 카테고리별 색상 배경 구분
cat_colors_bg = {"general": "#E3F2FD", "diet": "#E8F5E9", "research": "#FFF3E0", "review": "#FCE4EC"}
prev_cat = None
for i, (score, cat) in enumerate(zip(q_scores, q_cats)):
    color = ["#F44336", "#FFC107", "#8BC34A", "#4CAF50"][score]
    ax.bar(i, score, color=color, alpha=0.85, width=0.8)

ax.set_xticks(range(len(q_labels)))
ax.set_xticklabels(q_labels, rotation=90, fontsize=6)
ax.set_ylim(0, 3.5)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["0\n오답", "1\n부분", "2\n대부분", "3\n완전"])
ax.set_ylabel("품질 점수", fontsize=11)
ax.set_title(f"질문별 답변 품질 (카테고리별 정렬)\n{LABEL}", fontsize=12, fontweight="bold")

# 카테고리 경계선 + 레이블
prev = None
start_i = 0
for i, cat in enumerate(q_cats):
    if cat != prev:
        if prev is not None:
            ax.axvline(x=i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.text((start_i + i - 1) / 2, 3.3, prev, ha="center", fontsize=8,
                    color=cat_color_map.get(prev, "black"), fontweight="bold")
        start_i = i
        prev = cat
ax.text((start_i + len(q_cats) - 1) / 2, 3.3, prev, ha="center", fontsize=8,
        color=cat_color_map.get(prev, "black"), fontweight="bold")

# 평균선
ax.axhline(y=np.mean(q_scores), color="navy", linestyle="--", linewidth=1.2, alpha=0.7,
           label=f"평균 {np.mean(q_scores):.2f}")
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()
out = OUTPUT_DIR / "chart_qa6_per_question.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"저장: {out.name}")

print("\n모든 차트 생성 완료 (6개)")
