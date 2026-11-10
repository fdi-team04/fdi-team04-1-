"""1단계 후반: 학습 없는 '방향 미리 보기' — 변수 하나씩 5분위 수익률과 원시 IC
(판정용이 아님. 판정은 step3 XGBoost 단계별 IC와 step4 Fama-MacBeth)
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
OUT = "/home/user/fdi-team04-1-/연습장/out"
df = pd.read_parquet(f"{OUT}/features.parquet")
df = df[df.target.notna()]
VARS = {"beta": "베타", "size": "규모 ln(시총)", "bm": "가치 1/PBR", "rev": "단기 반전(t월 수익률)",
        "mom": "모멘텀(t-11~t-1)", "ivol": "고유변동성", "issue": "주식 발행",
        "retail_nb": "개인 순매수 비율", "foreign_nb": "외국인 순매수 비율"}

def tstat(x):
    x = pd.Series(x).dropna(); return x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))

rows, qrows, icm = [], [], {}
for v in VARS:
    d = df[["month", v, "target"]].dropna()
    ic = d.groupby("month").apply(lambda x: x[v].rank().corr(x.target.rank()) if len(x) >= 30 else np.nan,
                                  include_groups=False).dropna()
    icm[v] = ic
    d["q"] = d.groupby("month")[v].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    qm = d.groupby(["month", "q"]).target.mean().unstack()
    spread = qm[5] - qm[1]
    val = ic[ic.index >= "2011-01"]; sv = spread[spread.index >= "2011-01"]
    rows.append({"변수": VARS[v], "col": v, "원시IC_전체": ic.mean(), "t_전체": tstat(ic),
                 "원시IC_2011~21": val.mean(), "t_2011~21": tstat(val), "IC>0 비율(2011~21)": (val > 0).mean(),
                 "Q5-Q1(%p, 2011~21)": sv.mean() * 100, "t_Q5-Q1": tstat(sv), "시작": ic.index.min()})
    for q in range(1, 6):
        qrows.append({"col": v, "q": q, "전체": qm[q].mean() * 100,
                      "2011~21": qm.loc[qm.index >= "2011-01", q].mean() * 100})
res = pd.DataFrame(rows); qt = pd.DataFrame(qrows)
res.to_csv(f"{OUT}/preview_ic.csv", index=False, encoding="utf-8-sig")
qt.to_csv(f"{OUT}/preview_quintiles.csv", index=False, encoding="utf-8-sig")
print(res.round(4).to_string(index=False))

# 그림 A: 5분위 막대 (검증구간 2011~21)
fig, axes = plt.subplots(3, 3, figsize=(13, 10))
for ax, v in zip(axes.flat, VARS):
    s = qt[qt.col == v]
    ax.bar(s.q, s["2011~21"], color=["#c44e52" if q in (1, 5) else "#8c8c8c" for q in s.q])
    r = res[res.col == v].iloc[0]
    ax.set_title(f"{VARS[v]}\nQ5-Q1 {r['Q5-Q1(%p, 2011~21)']:+.2f}%p (t {r['t_Q5-Q1']:+.1f})", fontsize=10)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(["1\n낮음", "2", "3", "4", "5\n높음"])
    ax.set_ylabel("다음 달 평균 수익률(%)"); ax.axhline(0, color="k", lw=.5)
fig.suptitle("그림 A. 변수 5분위별 다음 달 평균 수익률 (2011~2021, 학습 없는 방향 미리 보기)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/figA_quintiles.png", dpi=130)

# 그림 B: 누적 원시 IC
fig, ax = plt.subplots(figsize=(12, 5.5))
for v in VARS:
    ic = icm[v]; ic = ic[ic.index >= "2011-01"]
    ax.plot(pd.to_datetime(ic.index), ic.cumsum(), label=VARS[v])
ax.axhline(0, color="k", lw=.5); ax.legend(ncol=3, fontsize=9); ax.set_ylabel("월별 IC 누적합")
ax.set_title("그림 B. 변수 하나씩 본 원시 IC 누적합 (2011~2021): 오른쪽 위로 가면 +, 아래로 가면 - 방향이 꾸준하다는 뜻")
fig.tight_layout(); fig.savefig(f"{OUT}/figB_cum_raw_ic.png", dpi=130)
