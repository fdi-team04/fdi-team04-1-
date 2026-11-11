"""발표용 추가 그래프: 9개 변수 5분위, 연도별 막대, H1 점검 t값"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = sys.argv[1]
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 16,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8C877D",
                     "axes.labelcolor": "#4A4740", "xtick.color": "#4A4740", "ytick.color": "#4A4740",
                     "figure.facecolor": "#FBFAF6", "axes.facecolor": "#FBFAF6", "savefig.facecolor": "#FBFAF6"})
INK, TEAL, RUST, GREY = "#1E2422", "#1F5C5A", "#B4532A", "#C9C3B6"
O = "/home/user/fdi-team04-1-/연습장/out"; S = f"{O}/slides"

# 1) 9개 변수 5분위
q = pd.read_csv(f"{O}/preview_quintiles.csv").pivot(index="col", columns="q", values="2011~21")
V = [("beta", "베타", "+", "시장과 따로", "크게 출렁"), ("size", "규모", "−", "작은 회사", "큰 회사"),
     ("bm", "가치 (1/PBR)", "+", "비싼 주식", "싼 주식"), ("rev", "단기 반전", "−", "이번 달 급락", "이번 달 급등"),
     ("mom", "모멘텀", "±", "1년 하락", "1년 상승"), ("ivol", "고유변동성", "−", "안정적", "불안정"),
     ("issue", "주식 발행", "−", "안 늘림", "많이 늘림"), ("retail_nb", "개인 순매수", "−", "많이 팜", "많이 삼"),
     ("foreign_nb", "외국인 순매수", "+", "많이 팜", "많이 삼")]
fig, axes = plt.subplots(3, 3, figsize=(16, 9.6), sharey=True)
for a, (c, nm, ex, lo, hi) in zip(axes.flat, V):
    v = q.loc[c].values
    good = {"+": bool(v[4] > v[0]), "−": bool(v[4] < v[0])}.get(ex)
    col = TEAL if good else (RUST if good is False else GREY)
    if c == "retail_nb": col = RUST
    a.bar(range(1, 6), v, color=[col if i in (1, 5) else GREY for i in range(1, 6)], width=0.62)
    a.set_title(f"{nm}   가설 {ex}", fontsize=18, color=INK, loc="left")
    a.set_xticks([1, 5]); a.set_xticklabels([lo, hi], fontsize=14)
    a.tick_params(axis="y", labelsize=13)
for a in axes[:, 0]: a.set_ylabel("다음 달 %", fontsize=14)
fig.tight_layout(h_pad=2.2); fig.savefig(f"{S}/c_quint9.png", dpi=150)

# 2) 연도별: 기준선 vs 수급까지, 막대 두 개
Y = pd.read_csv(f"{O}/stage_ic_by_year.csv", index_col=0)
b, f_ = Y["0 기준선"], Y["3 +수급(개인·외국인)"]
x = np.arange(len(Y)); fig, ax = plt.subplots(figsize=(14, 5.4))
ax.bar(x - 0.2, b, 0.4, color=GREY, label="기준선만")
ax.bar(x + 0.2, f_, 0.4, color=TEAL, label="수급까지 넣은 모델")
for i in range(len(x)):
    if f_.iloc[i] <= b.iloc[i] + 0.005: ax.text(x[i] + 0.2, f_.iloc[i] + 0.004, "비슷", ha="center", fontsize=13, color=RUST)
ax.set_xticks(x); ax.set_xticklabels(Y.index.astype(str), fontsize=15)
ax.set_ylabel("그해 평균 IC"); ax.legend(frameon=False, fontsize=15, loc="upper left")
fig.tight_layout(); fig.savefig(f"{S}/c_year2.png", dpi=150)
print("이긴 해(수급>기준선):", int((f_ > b).sum()))

# 3) H1 점검 t값
R = [("그대로", 3.14), ("극단 수익률\n1% 자름", 2.77), ("상장폐지\n−30%로 채움", 3.27), ("상장폐지\n−100%로 채움", 3.44),
     ("순매수 ±20%\n넘는 칸 제외", 3.22), ("수익률 0인\n칸 제외", 3.23)]
fig, ax = plt.subplots(figsize=(14, 5.4))
ax.bar(range(len(R)), [r[1] for r in R], color=RUST, width=0.58)
for i, (_, t) in enumerate(R): ax.text(i, t + 0.08, f"{t:.1f}", ha="center", fontsize=18, color=INK)
ax.axhline(2, color=INK, ls="--", lw=1.6); ax.text(-0.45, 1.75, "t = 2 (이 선 위면 우연으로 보기 어려움)", ha="left", fontsize=14, color=INK)
ax.set_xticks(range(len(R))); ax.set_xticklabels([r[0] for r in R], fontsize=14)
ax.set_ylim(0, 4); ax.set_ylabel("개인 순매수 t값")
fig.tight_layout(); fig.savefig(f"{S}/c_robust.png", dpi=150)
