"""학습 기간 비교: 확장(지금, 2001년부터 계속 쌓기) vs 롤링(최근 N년만)

왜 하나 (11/11 교수님 피드백 F9): "누적하는 건 OK인데 반대쪽에서 당길 필요는 없나… 학습을 꼭 다 써야
무조건 좋은 건 아닐 것 같다. 줄어드는 거 해서 조합은 돌려 놓고 확인만 하면 된다."
→ 오래된 해(예: 2001~2005)의 시장이 지금과 달라서 오히려 방해가 될 수 있는지 본다.
방법: 4단계(기준선/통제/보강/수급) × 학습 창 4종(확장, 롤링 10·7·5년) = 16번 돌림.
      Y년 예측 시 롤링 N년 = (Y−N)년 1월 ~ (Y−1)년 11월. 나머지 설정·평가(2011~2021 IC)는 동일.
주의: 결과를 보고 창을 고르는 게 아니라, "확장이 꼭 낫나"를 확인하는 점검이다. 본 판정 규칙은 그대로.
"""
import numpy as np, pandas as pd, sys
from xgboost import XGBRegressor
OUT = "/home/user/fdi-team04-1-/연습장/out"
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
df = pd.read_parquet(f"{OUT}/features.parquet"); df = df[df.month >= "2001-01"].copy()
for c in ALL:
    df[c + "_m"] = df[c].isna().astype(float)
    df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
df["y"] = df.groupby("month").target.rank(pct=True)
STAGES = [("기준선", ["beta", "size", "bm"]), ("+통제", ["rev", "mom"]), ("+보강", ["ivol", "issue"]), ("+수급", ["retail_nb", "foreign_nb"])]
WIN = [("확장", None), ("롤링10년", 10), ("롤링7년", 7), ("롤링5년", 5)]

def feats(cols):
    return [c + "_r" for c in cols] + [c + "_m" for c in cols if c in ("beta", "bm", "mom", "ivol", "issue")]

rows, ics = [], {}
for wname, N in WIN:
    cols = []
    for sname, add in STAGES:
        cols = cols + add; F = feats(cols); out = []
        for Y in range(2011, 2022):
            start = "2001-01" if N is None else f"{max(2001, Y - N)}-01"
            tr = df[(df.month >= start) & (df.month <= f"{Y-1}-11") & df.y.notna()]
            te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
            m = XGBRegressor(**XGB).fit(tr[F], tr.y)
            out.append(te[["month", "target"]].assign(score=m.predict(te[F])))
        p = pd.concat(out)
        ic = p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
        ics[(wname, sname)] = ic
        rows.append({"학습창": wname, "단계": sname, "평균IC": ic.mean(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic)))})
        print(rows[-1], flush=True)
R = pd.DataFrame(rows)
base = {s: ics[("확장", s)] for _, s in [(0, x[0]) for x in STAGES]}
R["확장 대비"] = [r.평균IC - ics[("확장", r.단계)].mean() for r in R.itertuples()]
R["확장보다 나은 해"] = [int((ics[(r.학습창, r.단계)].groupby(ics[(r.학습창, r.단계)].index.str[:4]).mean() >
                          ics[("확장", r.단계)].groupby(ics[("확장", r.단계)].index.str[:4]).mean()).sum()) for r in R.itertuples()]
R.to_csv(f"{OUT}/rolling_vs_expanding.csv", index=False, encoding="utf-8-sig")
pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in ics.items()}).to_csv(f"{OUT}/rolling_ic_monthly.csv", encoding="utf-8-sig")
print(R.round(4).to_string(index=False))
