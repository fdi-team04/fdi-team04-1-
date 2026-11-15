"""S3: 9개 변수 5분위를 기간별로 나눠 보기 + 학습창 비교 그림

왜 하나 (11/11 피드백): "전 기간 평균만 보면 어느 시기에 생긴 효과인지 모른다" → 2001–10 / 2011–15 / 2016–21로 나눠
같은 방향이 계속 유지되는지 본다. 동일가중, 매월 5분위 → Q5−Q1 월평균, t = 평균/(표준편차/√월수).
같이: step13 결과(확장 vs 롤링)를 그림 하나로.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 13,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = "out"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
NM = {"beta": "베타", "size": "규모", "bm": "가치", "rev": "단기 반전", "mom": "모멘텀", "ivol": "고유변동성",
      "issue": "주식 발행", "retail_nb": "개인 순매수", "foreign_nb": "외국인 순매수"}
df = pd.read_parquet(f"{OUT}/features.parquet"); df = df[(df.month >= "2001-01") & df.target.notna()]
tg = df.target * (100 if df.target.abs().median() < 1 else 1)
df = df.assign(t=tg)
P = [("2001–2010", "2001-01", "2010-12"), ("2011–2015", "2011-01", "2015-12"), ("2016–2021", "2016-01", "2021-12")]
rows = []
for c in ALL:
    x = df[df[c].notna()].copy()
    x["q"] = x.groupby("month")[c].transform(lambda v: pd.qcut(v.rank(method="first"), 5, labels=False) + 1)
    m = x.groupby(["month", "q"]).t.mean().unstack()
    for nm, a, b in P:
        s = m[(m.index >= a) & (m.index <= b)]
        d = (s[5] - s[1]).dropna()
        rows.append({"col": c, "변수": NM[c], "기간": nm, **{f"Q{i}": s[i].mean() for i in range(1, 6)},
                     "Q5-Q1(%p)": d.mean(), "t": d.mean() / (d.std() / np.sqrt(len(d))), "월수": len(d)})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/quint_by_period.csv", index=False, encoding="utf-8-sig")
W = R.pivot(index="변수", columns="기간", values="Q5-Q1(%p)").loc[[NM[c] for c in ALL]]
T = R.pivot(index="변수", columns="기간", values="t").loc[[NM[c] for c in ALL]]
print(W.round(2).astype(str) + " (t " + T.round(1).astype(str) + ")")
fig, ax = plt.subplots(figsize=(13, 6)); x = np.arange(len(W)); cols = ["#C9C3B6", "#6E9C99", "#1F5C5A"]
for i, p in enumerate(W.columns):
    ax.bar(x + (i - 1) * 0.27, W[p], 0.27, color=cols[i], label=p)
    for j, (v, t) in enumerate(zip(W[p], T[p])):
        if abs(t) >= 2: ax.text(x[j] + (i - 1) * 0.27, v + (0.06 if v >= 0 else -0.16), "*", ha="center", fontsize=14)
ax.axhline(0, color="#8C877D", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(W.index)
ax.set_ylabel("Q5−Q1 다음 달 수익률 차 (%p/월)"); ax.legend(frameon=False)
ax.set_title("변수별 상·하위 20% 수익률 차이, 기간별 (* = |t|≥2)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figM_quint_by_period.png", dpi=140)

r = pd.read_csv(f"{OUT}/rolling_vs_expanding.csv")
fig, ax = plt.subplots(figsize=(11, 5)); st = ["기준선", "+통제", "+보강", "+수급"]; x = np.arange(4)
for i, (w, c) in enumerate(zip(["확장", "롤링10년", "롤링7년", "롤링5년"], ["#1F5C5A", "#6E9C99", "#A9C2BF", "#C9C3B6"])):
    v = r[r.학습창 == w].set_index("단계").loc[st, "평균IC"]
    ax.bar(x + (i - 1.5) * 0.2, v, 0.2, color=c, label=w)
ax.set_xticks(x); ax.set_xticklabels(st); ax.set_ylabel("평균 IC (2011–2021)"); ax.legend(frameon=False, ncol=4)
ax.set_title("학습 창 비교: 모든 단계에서 확장(2001년부터 누적)이 가장 높음", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figN_rolling.png", dpi=140)
