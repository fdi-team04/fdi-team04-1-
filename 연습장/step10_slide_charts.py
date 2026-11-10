"""발표용 그래프 (제목 없이, 슬라이드 테마 색·글꼴에 맞춤)"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = sys.argv[1]
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 17,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8C877D",
                     "axes.labelcolor": "#4A4740", "xtick.color": "#4A4740", "ytick.color": "#4A4740",
                     "figure.facecolor": "#FBFAF6", "axes.facecolor": "#FBFAF6", "savefig.facecolor": "#FBFAF6"})
INK, TEAL, RUST, GREY = "#1E2422", "#1F5C5A", "#B4532A", "#C9C3B6"
O = "/home/user/fdi-team04-1-/연습장/out"; S = f"{O}/slides"; import os; os.makedirs(S, exist_ok=True)
R = pd.read_csv(f"{O}/stage_ic.csv").set_index("단계")

def bars(names, labels, fn, w=12, h=5.6):
    r = R.loc[names]
    fig, ax = plt.subplots(figsize=(w, h))
    col = [GREY if pd.isna(v) else (TEAL if v == "OK" else RUST) for v in r.판정]
    ax.bar(range(len(r)), r.평균IC, color=col, width=0.62)
    for i, (n, row) in enumerate(r.iterrows()):
        ax.text(i, row.평균IC + 0.0025, f"{row.평균IC:.3f}", ha="center", fontsize=19, color=INK, fontweight="bold")
        if not pd.isna(row.판정):
            ax.text(i, row.평균IC / 2, f"{row.증가분:+.3f}\n{row['이긴 해']}", ha="center", va="center", fontsize=15, color="#FBFAF6")
    ax.set_xticks(range(len(r))); ax.set_xticklabels(labels, fontsize=17)
    ax.set_ylim(0, r.평균IC.max() * 1.18); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{S}/{fn}", dpi=150)

bars(["0 기준선", "1 +통제(반전·모멘텀)", "2 +보강(고유변동성·발행)", "3 +수급(개인·외국인)"],
     ["기준선", "+ 통제", "+ 보강", "+ 수급"], "c_stage.png")
bars(["s1 베타", "s2 +규모", "s3 +가치", "s4 +모멘텀", "s5 +단기반전", "s6 +고유변동성", "s7 +주식발행"],
     ["베타", "+규모", "+가치", "+모멘텀", "+단기반전", "+고유변동성", "+주식발행"], "c_seq.png", w=13)

# 연도별: 기준선 vs 수급까지
Y = pd.read_csv(f"{O}/stage_ic_by_year.csv", index_col=0)
fig, ax = plt.subplots(figsize=(12, 5.2))
x = Y.index.astype(str)
ax.plot(x, Y["0 기준선"], color=GREY, marker="o", lw=3, ms=8, label="기준선")
ax.plot(x, Y["1 +통제(반전·모멘텀)"], color=TEAL, marker="o", lw=2, ms=6, alpha=0.55, label="+ 통제")
ax.plot(x, Y["3 +수급(개인·외국인)"], color=TEAL, marker="o", lw=3.5, ms=9, label="+ 수급까지")
ax.axhline(0, color="#8C877D", lw=0.8); ax.legend(frameon=False, fontsize=16, loc="upper left")
ax.set_ylabel("그해 평균 IC"); fig.tight_layout(); fig.savefig(f"{S}/c_year.png", dpi=150)

# 누적 IC + 누수 점검
M = pd.read_csv(f"{O}/stage_ic_monthly.csv", index_col=0); t = pd.to_datetime(M.index)
fig, ax = plt.subplots(figsize=(12, 5.2))
ax.plot(t, M["0 기준선"].cumsum(), color=GREY, lw=3, label="기준선")
ax.plot(t, M["3 +수급(개인·외국인)"].cumsum(), color=TEAL, lw=3.5, label="+ 수급까지")
ax.plot(t, M["누수점검(정답 섞기)"].cumsum(), color=RUST, lw=2.5, ls="--", label="정답을 섞어 학습 (누수 점검)")
ax.axhline(0, color="#8C877D", lw=0.8); ax.legend(frameon=False, fontsize=16, loc="upper left"); ax.set_ylabel("월별 IC 누적합")
fig.tight_layout(); fig.savefig(f"{S}/c_cum.png", dpi=150)

# H1·H2 5분위 (전체)
Q = pd.read_csv(f"{O}/h1h2_quintile.csv"); Q = Q[Q.그룹 == "전체"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
for a, (_, r), c, lab in zip(axes, Q.iterrows(), [RUST, TEAL], ["개인 순매수", "외국인 순매수"]):
    v = [r[f"Q{q}"] for q in range(1, 6)]
    a.bar(range(1, 6), v, color=[c if q in (1, 5) else GREY for q in range(1, 6)], width=0.62)
    for q, val in zip(range(1, 6), v): a.text(q, val + 0.03, f"{val:.2f}", ha="center", fontsize=16, color=INK)
    a.set_xticks(range(1, 6)); a.set_xticklabels(["많이 팜", "2", "3", "4", "많이 삼"], fontsize=16)
    a.set_title(f"{lab}   5−1 = {r['Q5-Q1(%p)']:+.2f}%p  (t {r['t']:+.1f})", fontsize=18, color=INK, loc="left")
axes[0].set_ylabel("다음 달 평균 수익률 (%)")
fig.tight_layout(); fig.savefig(f"{S}/c_h12.png", dpi=150)

# Fama-MacBeth t
F = pd.read_csv(f"{O}/fama_macbeth.csv"); F = F[F.구간 == "검증 2011~2021"]
nm = {"beta": "베타", "size": "규모", "bm": "가치", "rev": "단기 반전", "mom": "모멘텀", "ivol": "고유변동성",
      "issue": "주식 발행", "retail_nb": "개인 순매수", "foreign_nb": "외국인 순매수"}
fig, ax = plt.subplots(figsize=(12, 6))
cols = [TEAL if "지지" in v else RUST if "반대" in v else GREY for v in F.판정]
y = np.arange(len(F))[::-1]
ax.barh(y, F["t"], color=cols, height=0.62)
for yy, tv, ex in zip(y, F["t"], F["예상"]):
    ax.text(tv + (0.15 if tv >= 0 else -0.15), yy, f"{tv:+.1f}   (예상 {ex})", va="center", ha="left" if tv >= 0 else "right", fontsize=15, color=INK)
ax.set_yticks(y); ax.set_yticklabels([nm[c] for c in F.col], fontsize=17)
for xv in (-2, 2): ax.axvline(xv, color="#8C877D", ls=":", lw=1.4)
ax.axvline(0, color="#8C877D", lw=0.8); ax.set_xlim(-8.5, 6.5); ax.set_xlabel("t값  (점선 ±2)")
fig.tight_layout(); fig.savefig(f"{S}/c_fm.png", dpi=150)

# 데이터: 가치 변수 채움 비율 + 종목 수
f = pd.read_parquet(f"{O}/features.parquet")
fill = f.groupby("month").bm.apply(lambda s: s.notna().mean() * 100); t = pd.to_datetime(fill.index)
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.fill_between(t, fill, color=TEAL, alpha=0.18); ax.plot(t, fill, color=TEAL, lw=2.5)
ax.axvspan(pd.Timestamp("2011-01"), pd.Timestamp("2021-12"), color="#E9E4D8", zorder=0)
ax.annotate("2002년 4월 전\nPBR 없음", (pd.Timestamp("2001-06"), 3), xytext=(pd.Timestamp("2003-06"), 35), fontsize=15, arrowprops=dict(arrowstyle="-", color=INK))
ax.annotate("2008년 8~11월\n통째로 없음", (pd.Timestamp("2008-10"), 3), xytext=(pd.Timestamp("2009-06"), 35), fontsize=15, arrowprops=dict(arrowstyle="-", color=INK))
ax.text(pd.Timestamp("2014-06"), 50, "검증 구간\n빈칸 0.9%", fontsize=16, color=INK)
ax.set_ylim(0, 105); ax.set_ylabel("가치 변수가\n있는 종목 (%)")
fig.tight_layout(); fig.savefig(f"{S}/c_pbr.png", dpi=150)
print("ok")
