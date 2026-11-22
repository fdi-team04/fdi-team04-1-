"""새 H4: 목표가 상향 기사가 난 종목 중 개인이 많이 산 종목은 다음 달 효과가 약하다 (원래 H4 '호재 + 개인 매수 → 약함'을 새 주제로)
표본: 전년 12월 말 시총 상위 200 (out/목표가상향_변수_합치기용.csv), 2001–2021, 다음 달 수익률
개인 매수: 같은 달(t월) 개인 순매수 / 시총 (retail_nb), 매달 순위
판정 기준(결과 보기 전에 정함, 11/23):
  ① 차이의 차이: [목표가 종목 안에서 개인 많이 − 적게] − [나머지 종목 안에서 같은 차이] 가 음수이고 t <= -2
  ② 또는 통제 9개 FM에 tp_up × 개인순매수순위 상호작용을 넣었을 때 계수 t <= -2
  그리고 ③ 2016–21에도 음수 → 지지 / 아니면 지지 안 됨
"""
import numpy as np, pandas as pd
O = "/home/user/fdi-team04-1-/연습장/out_yw"; ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
nw = pd.read_csv("/home/user/fdi-team04-1-/연습장/out/목표가상향_변수_합치기용.csv", dtype={"code": str})
u = pd.read_parquet(f"{O}/features.parquet").merge(nw[["month", "code", "tp_up"]], on=["month", "code"])
u = u[u.target.notna() & u.retail_nb.notna()].copy()
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
u["rhi"] = (u.groupby(["month", "tp_up"]).retail_nb.rank(pct=True) > 0.5).astype(int)
g = u.groupby(["month", "tp_up", "rhi"]).target.mean().unstack(["tp_up", "rhi"]) * 100
dg = (g[(1, 1)] - g[(1, 0)]).dropna(); do = (g[(0, 1)] - g[(0, 0)]).dropna(); dd = (dg - do).dropna()
P = [("전체", "2001", "2021-12"), ("2001–10", "2001", "2010-12"), ("2011–15", "2011", "2015-12"), ("2016–21", "2016", "2021-12")]
cut = lambda s, a, b: s[(s.index >= a) & (s.index <= b)]
rows = [{"기간": p, "목표가 종목: 개인 많이−적게": cut(dg, a, b).mean(), "t1": tt(cut(dg, a, b)), "나머지: 개인 많이−적게": cut(do, a, b).mean(), "t2": tt(cut(do, a, b)),
         "차이의 차이": cut(dd, a, b).mean(), "t": tt(cut(dd, a, b))} for p, a, b in P]
u["inter"] = u.tp_up * u.retail_nb_r; Bm = {}
for p, a, b in P:
    B = []
    for m, x in u[(u.month >= a) & (u.month <= b)].groupby("month"):
        if len(x) < 30 or x.tp_up.sum() < 3: continue
        X = np.column_stack([np.ones(len(x))] + [x[c + "_r"] for c in ALL] + [x.tp_up, x.inter]); B.append(np.linalg.lstsq(X, x.target * 100, rcond=None)[0][-2:])
    B = pd.DataFrame(B, columns=["tp_up", "inter"]); Bm[p] = (B.inter.mean(), tt(B.inter), B.tp_up.mean(), tt(B.tp_up))
R = pd.DataFrame(rows); R["FM 상호작용"] = [Bm[p][0] for p, _, _ in P]; R["FM 상호작용 t"] = [Bm[p][1] for p, _, _ in P]
R.to_csv(f"{O}/h4_tp_retail.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.round(3).to_string(index=False))
a = R.iloc[0]; rec = R.iloc[3]
ok = (a["차이의 차이"] < 0 and a.t <= -2) or (a["FM 상호작용"] < 0 and a["FM 상호작용 t"] <= -2)
print("판정:", "지지" if ok and (rec["차이의 차이"] < 0 or rec["FM 상호작용"] < 0) else "지지 안 됨")
