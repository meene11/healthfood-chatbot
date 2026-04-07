"""
실험 8: 3개 모델 비교 시각화 스크립트
GPT-4o-mini vs Llama 3.3 70B vs Gemini Flash Lite
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import json, os
from pathlib import Path

def set_korean_font():
    for font in ["Malgun Gothic", "맑은 고딕", "NanumGothic", "AppleGothic"]:
        if font in [f.name for f in fm.fontManager.ttflist]:
            plt.rcParams["font.family"] = font
            break
    plt.rcParams["axes.unicode_minus"] = False

set_korean_font()
OUTPUT_DIR = Path(__file__).parent

def load_results():
    results = []
    for f in sorted(OUTPUT_DIR.glob("eval_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            results.append(d)
        except:
            pass
    order = {"gpt-4o-mini": 0, "groq-llama": 1, "gemini-flash": 2}
    results.sort(key=lambda x: order.get(x["model"], 9))
    return results

data = load_results()
print(f"로드된 모델: {[d['label'] for d in data]}")

labels_short = ["GPT-4o-mini", "Llama 3.3 70B\n(Groq 무료)", "Gemini Flash\nLite (무료)"]
qualities  = [d["avg_quality"] for d in data]
coverages  = [d["avg_coverage"] for d in data]
times      = [d["avg_time"] for d in data]
n          = data[0]["n_valid"]

dist    = [{int(k): v for k, v in d["quality_dist"].items()} for d in data]
perfect = [dd.get(3, 0) for dd in dist]
mostly  = [dd.get(2, 0) for dd in dist]
partial = [dd.get(1, 0) for dd in dist]
wrong   = [dd.get(0, 0) for dd in dist]

MODEL_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
x = np.arange(len(data))

# ── 차트 1: 평균 품질 + 완전정답률 ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

bars = axes[0].bar(x, qualities, color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_short, fontsize=9)
axes[0].set_ylim(0, 3.0); axes[0].set_ylabel("평균 품질 점수 (0~3)", fontsize=11)
axes[0].set_title("모델별 평균 답변 품질\n(LLM-as-Judge, 62문항)", fontsize=12, fontweight="bold")
axes[0].yaxis.grid(True, alpha=0.3); axes[0].set_axisbelow(True)
for bar, v in zip(bars, qualities):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.03, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
bars[qualities.index(max(qualities))].set_edgecolor("gold"); bars[qualities.index(max(qualities))].set_linewidth(2.5)

pcts = [p/n*100 for p in perfect]
bars2 = axes[1].bar(x, pcts, color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels(labels_short, fontsize=9)
axes[1].set_ylim(0, 55); axes[1].set_ylabel("완전 정답 비율 (%)", fontsize=11)
axes[1].set_title("모델별 완전 정답률 (3점 비율)\n(높을수록 좋음)", fontsize=12, fontweight="bold")
axes[1].yaxis.grid(True, alpha=0.3); axes[1].set_axisbelow(True)
for bar, v, cnt in zip(bars2, pcts, perfect):
    axes[1].text(bar.get_x()+bar.get_width()/2, v+0.8, f"{v:.1f}%\n({cnt}개)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
bars2[pcts.index(max(pcts))].set_edgecolor("gold"); bars2[pcts.index(max(pcts))].set_linewidth(2.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp1_quality.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp1_quality.png")

# ── 차트 2: 품질 분포 누적 바 ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
w_pct = [v/n*100 for v in wrong]
p_pct = [v/n*100 for v in partial]
m_pct = [v/n*100 for v in mostly]
f_pct = [v/n*100 for v in perfect]

ax.bar(x, w_pct, color="#F44336", alpha=0.85, width=0.5, label="0점 (오답/모름)")
ax.bar(x, p_pct, color="#FFC107", alpha=0.85, width=0.5, label="1점 (부분정답)", bottom=w_pct)
ax.bar(x, m_pct, color="#8BC34A", alpha=0.85, width=0.5, label="2점 (대부분정확)",
       bottom=[a+b for a,b in zip(w_pct, p_pct)])
ax.bar(x, f_pct, color="#4CAF50", alpha=0.85, width=0.5, label="3점 (완전정답)",
       bottom=[a+b+c for a,b,c in zip(w_pct, p_pct, m_pct)])

ax.set_xticks(x); ax.set_xticklabels(labels_short, fontsize=10)
ax.set_ylim(0, 105); ax.set_ylabel("비율 (%)", fontsize=11)
ax.set_title("모델별 답변 품질 분포 (누적 100%)\n(초록 많을수록 좋음)", fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)

for i in range(len(data)):
    y3 = w_pct[i] + p_pct[i] + m_pct[i] + f_pct[i]/2
    y2 = w_pct[i] + p_pct[i] + m_pct[i]/2
    y1 = w_pct[i] + p_pct[i]/2
    y0 = w_pct[i]/2
    for y, v, cnt in [(y0,w_pct[i],wrong[i]),(y1,p_pct[i],partial[i]),(y2,m_pct[i],mostly[i]),(y3,f_pct[i],perfect[i])]:
        if v > 5:
            ax.text(i, y, f"{v:.0f}%\n({cnt})", ha="center", va="center", fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp2_distribution.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp2_distribution.png")

# ── 차트 3: 레이더 차트 ───────────────────────────────────────────────
categories_radar = ["평균 품질\n(÷3)", "완전정답률", "오답 회피율\n(1-오답률)", "응답 속도\n(역수)", "검색커버리지"]
N_r = len(categories_radar)
angles = [n2 / float(N_r) * 2 * np.pi for n2 in range(N_r)] + [0]
max_time = max(times)

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for d_item, t, label, color in zip(data, times, labels_short, MODEL_COLORS):
    dj = {int(k): v for k, v in d_item["quality_dist"].items()}
    vals = [
        d_item["avg_quality"] / 3.0,
        dj.get(3,0) / n,
        1 - dj.get(0,0) / n,
        1 - (t / max_time),
        d_item["avg_coverage"],
    ]
    vals_plot = vals + [vals[0]]
    ax.plot(angles, vals_plot, color=color, linewidth=2, label=label.replace("\n"," "))
    ax.fill(angles, vals_plot, color=color, alpha=0.15)

ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories_radar, fontsize=10)
ax.set_ylim(0, 1); ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
ax.set_title("모델 종합 비교 레이더 차트\n(모든 축: 높을수록 좋음)", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp3_radar.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp3_radar.png")

# ── 차트 4: 카테고리별 히트맵 ────────────────────────────────────────
cats = ["diet","general","research","review"]
cat_labels = ["다이어트\n(n=16)","일반건강\n(n=40)","연구방법\n(n=4)","리뷰\n(n=2)"]
hm = np.array([[d["cat_avg"].get(c,0) for c in cats] for d in data])

fig, ax = plt.subplots(figsize=(9, 4))
im = ax.imshow(hm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=3)
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cat_labels, fontsize=10)
ax.set_yticks(range(len(data))); ax.set_yticklabels(labels_short, fontsize=9)
ax.set_title("카테고리 x 모델 품질 히트맵\n(녹색=높음, 빨강=낮음)", fontsize=12, fontweight="bold")
for i in range(len(data)):
    for j in range(len(cats)):
        v = hm[i,j]
        color = "white" if v < 0.8 or v > 2.3 else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12, color=color, fontweight="bold")
plt.colorbar(im, ax=ax, shrink=0.8, label="평균 품질 (0~3)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp4_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp4_heatmap.png")

# ── 차트 5: 응답 시간 + 비용 대비 성능 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

bars = axes[0].bar(x, times, color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_short, fontsize=9)
axes[0].set_ylabel("평균 응답 시간 (초)", fontsize=11)
axes[0].set_title("모델별 평균 응답 시간\n(낮을수록 좋음)", fontsize=12, fontweight="bold")
axes[0].yaxis.grid(True, alpha=0.3); axes[0].set_axisbelow(True)
for bar, v in zip(bars, times):
    axes[0].text(bar.get_x()+bar.get_width()/2, v+0.1, f"{v:.1f}초",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")
bars[times.index(min(times))].set_edgecolor("gold"); bars[times.index(min(times))].set_linewidth(2.5)

costs = [0.15, 0.0, 0.0]
axes[1].scatter(costs, qualities, c=MODEL_COLORS, s=300, zorder=5, alpha=0.9)
annots = [("GPT-4o-mini\n$0.15/1M\n품질: 1.581", 0.02, 0.03),
          ("Llama 3.3 70B\n무료(Groq)\n품질: 1.629", -0.01, -0.07),
          ("Gemini Flash\n무료(Google)\n품질: 1.468", -0.01, 0.05)]
for (lbl, dx, dy), (cx, cy) in zip(annots, zip(costs, qualities)):
    axes[1].annotate(lbl, xy=(cx,cy), xytext=(cx+dx, cy+dy), fontsize=8.5,
                     ha="center", arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
axes[1].set_xlabel("API 비용 ($/1M 입력 토큰)", fontsize=11)
axes[1].set_ylabel("평균 답변 품질 (0~3)", fontsize=11)
axes[1].set_xlim(-0.05, 0.25); axes[1].set_ylim(1.3, 1.8)
axes[1].set_title("비용 대비 성능\n(왼쪽 상단 = 이상적)", fontsize=12, fontweight="bold")
axes[1].xaxis.grid(True, alpha=0.3); axes[1].yaxis.grid(True, alpha=0.3); axes[1].set_axisbelow(True)
axes[1].axvline(x=0.02, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp5_cost_perf.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp5_cost_perf.png")

# ── 차트 6: 종합 대시보드 ────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("실험 8: 모델 비교 평가 종합 대시보드\n(GPT-4o-mini vs Llama 3.3 70B vs Gemini Flash Lite | 62문항)",
             fontsize=14, fontweight="bold")

short3 = ["GPT\n4o-mini","Llama\n3.3 70B","Gemini\nFlash"]

axes[0,0].bar(x, qualities, color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[0,0].set_xticks(x); axes[0,0].set_xticklabels(short3, fontsize=9)
axes[0,0].set_ylim(0,2.5); axes[0,0].set_title("평균 품질 (0~3)", fontweight="bold")
for i,v in enumerate(qualities): axes[0,0].text(i, v+0.03, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
axes[0,0].yaxis.grid(True, alpha=0.3); axes[0,0].set_axisbelow(True)

axes[0,1].bar(x, [p/n*100 for p in perfect], color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[0,1].set_xticks(x); axes[0,1].set_xticklabels(short3, fontsize=9)
axes[0,1].set_ylim(0,50); axes[0,1].set_title("완전정답률 % (3점)", fontweight="bold")
for i,v in enumerate([p/n*100 for p in perfect]): axes[0,1].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[0,1].yaxis.grid(True, alpha=0.3); axes[0,1].set_axisbelow(True)

axes[0,2].bar(x, times, color=MODEL_COLORS, alpha=0.85, width=0.5)
axes[0,2].set_xticks(x); axes[0,2].set_xticklabels(short3, fontsize=9)
axes[0,2].set_title("평균 응답 시간 (초)", fontweight="bold")
for i,v in enumerate(times): axes[0,2].text(i, v+0.1, f"{v:.1f}s", ha="center", fontsize=10, fontweight="bold")
axes[0,2].yaxis.grid(True, alpha=0.3); axes[0,2].set_axisbelow(True)

axes[1,0].bar(x, [v/n*100 for v in wrong], color=["#E57373","#EF9A9A","#FFCDD2"], alpha=0.85, width=0.5)
axes[1,0].set_xticks(x); axes[1,0].set_xticklabels(short3, fontsize=9)
axes[1,0].set_ylim(0,50); axes[1,0].set_title("오답/모름 비율 % (낮을수록 좋음)", fontweight="bold")
for i,v in enumerate([v/n*100 for v in wrong]): axes[1,0].text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=10, fontweight="bold")
axes[1,0].yaxis.grid(True, alpha=0.3); axes[1,0].set_axisbelow(True)

axes[1,1].bar(x, w_pct, color="#F44336", alpha=0.85, width=0.5, label="0점")
axes[1,1].bar(x, p_pct, color="#FFC107", alpha=0.85, width=0.5, label="1점", bottom=w_pct)
axes[1,1].bar(x, m_pct, color="#8BC34A", alpha=0.85, width=0.5, label="2점", bottom=[a+b for a,b in zip(w_pct,p_pct)])
axes[1,1].bar(x, f_pct, color="#4CAF50", alpha=0.85, width=0.5, label="3점", bottom=[a+b+c for a,b,c in zip(w_pct,p_pct,m_pct)])
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(short3, fontsize=9)
axes[1,1].set_title("품질 분포 (누적)", fontweight="bold")
axes[1,1].legend(fontsize=8, loc="upper right")
axes[1,1].yaxis.grid(True, alpha=0.3); axes[1,1].set_axisbelow(True)

axes[1,2].axis("off")
tbl_data = [
    ["지표", "GPT-4o\nmini", "Llama\n3.3 70B", "Gemini\nFlash"],
    ["평균 품질", "1.581", "★1.629", "1.468"],
    ["완전정답", "★35.5%", "27.4%", "33.9%"],
    ["오답률",   "33.9%", "★25.8%", "35.5%"],
    ["응답시간", "8.1초", "★5.6초", "14.0초"],
    ["비용",     "$0.15/1M", "★무료", "★무료"],
]
tbl = axes[1,2].table(cellText=tbl_data[1:], colLabels=tbl_data[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.2, 1.7)
for j in range(4):
    tbl[(0,j)].set_facecolor("#37474F"); tbl[(0,j)].set_text_props(color="white", fontweight="bold")
axes[1,2].set_title("종합 비교표 (★ = 1위)", fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR/"chart_cmp6_dashboard.png", dpi=150, bbox_inches="tight")
plt.close(); print("저장: chart_cmp6_dashboard.png")

print("\n모든 차트 생성 완료 (6개)")
