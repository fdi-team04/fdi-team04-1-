"""2단계 그림: 단계별 IC 막대, 연도별 IC, 하나씩 쌓기, 누적 IC"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
OUT = "/home/user/fdi-team04-1-/연습장/out"
R = pd.read_csv(f"{OUT}/stage_ic.csv"); Y = pd.read_csv(f"{OUT}/stage_ic_by_year.csv", index_col=0)
M = pd.read_csv(f"{OUT}/stage_ic_monthly.csv", index_col=0)
main = ["0 기준선", "1 +통제(반전·모멘텀)", "2 +보강(고유변동성·발행)", "3 +수급(개인·외국인)"]
seq = ["s1 베타", "s2 +규모", "s3 +가치", "s4 +모멘텀", "s5 +단기반전", "s6 +고유변동성", "s7 +주식발행"]

def bars(names, fname, title):
    r = R.set_index("단계").loc[names]
    fig, ax = plt.subplots(figsize=(11, 5))
    col = ["#4c72b0" if pd.isna(v) else ("#55a868" if v == "OK" else "#c44e52") for v in r.판정]
    ax.bar(range(len(r)), r.평균IC, color=col)
    for i, (n, row) in enumerate(r.iterrows()):
        lab = f"{row.평균IC:.4f}\n(t {row.t:.1f})"
        if not pd.isna(row.판정): lab += f"\n{row.증가분:+.4f}, {row['이긴 해']}\n{row.판정}"
        ax.text(i, row.평균IC + 0.002, lab, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(r))); ax.set_xticklabels([n.replace(" +", "\n+") for n in r.index], fontsize=9)
    ax.set_ylim(0, r.평균IC.max() * 1.45); ax.set_ylabel("평균 IC (2011~2021, 131개월)")
    ax.set_title(title + "\n초록=OK(평균IC↑ & 11년 중 6년↑), 빨강=NO, 파랑=출발점")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}", dpi=130)

bars(main, "figC_stage_ic.png", "그림 C. XGBoost 누적 학습 단계별 평균 IC")
bars(seq, "figD_seq_ic.png", "그림 D. 변수를 하나씩 쌓을 때 평균 IC (XGBoost)")

fig, ax = plt.subplots(figsize=(12, 5))
for n, c in zip(main, ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]):
    ax.plot(Y.index.astype(str), Y[n], marker="o", label=n, color=c)
ax.axhline(0, color="k", lw=.5); ax.legend(); ax.set_ylabel("연평균 IC")
ax.set_title("그림 C2. 단계별 연도별 평균 IC: 선이 위로 올라간 해 = 그 단계가 '이긴 해'")
fig.tight_layout(); fig.savefig(f"{OUT}/figC2_stage_ic_by_year.png", dpi=130)

fig, ax = plt.subplots(figsize=(12, 5))
for n, c in zip(main + ["누수점검(정답 섞기)"], ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868", "#000000"]):
    ax.plot(pd.to_datetime(M.index), M[n].cumsum(), label=n, color=c, ls="--" if "누수" in n else "-")
ax.axhline(0, color="k", lw=.5); ax.legend(); ax.set_ylabel("월별 IC 누적합")
ax.set_title("그림 C3. 단계별 누적 IC (2011~2021) — 점선: 정답을 섞어 학습한 누수 점검(0 근처여야 정상)")
fig.tight_layout(); fig.savefig(f"{OUT}/figC3_stage_cum_ic.png", dpi=130)
