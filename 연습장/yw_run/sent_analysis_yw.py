"""전체 기사 감성 점수(sent_monthly.csv, 129만 건) → 같은 달 / 다음 달 수익률. 발표 9장(파일럿 52개 파일)을 전체로 다시 확인.
표본: ESG 분석과 같음 (해마다 직전 12월 시총 상위 200 × 그해 기사 1건 이상, 팀 데이터셋 수익률)
변수: s_all 전체 논조, s_noprice 주가 보도 뺀 논조, pos/neg_share 강한 긍정·부정 비율, surprise 논조 변화(직전 12달 평균 대비), log_n 기사 수
"""
import sys, os, json, numpy as np, pandas as pd
from xgboost import XGBRegressor
O = "/home/user/fdi-team04-1-/연습장/out_yw"; P = f"{O}/news_sent_full"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
R = {}
nm = pd.read_csv(f"{P}/sent_monthly_input.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
nm = nm.sort_values(["code", "month"])
nm["surprise"] = nm.s_all - nm.groupby("code").s_all.transform(lambda s: s.shift(1).rolling(12, min_periods=3).mean())
nm["log_n"] = np.log1p(nm.n)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
f["t2"] = f.groupby("code").target.shift(-1)
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left")
u["n"] = u.n.fillna(0); u["log_n"] = u.log_n.fillna(0)
u = u[(u.groupby(["code", "year"]).n.transform("sum") > 0) & u.target.notna() & (u.month >= "2001-01")].copy()
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
u["t2_r"] = u.groupby("month").t2.rank(pct=True)
R["표본"] = {"종목-월": len(u), "기사 있는 종목-월": int(u.s_all.notna().sum()), "달": int(u.month.nunique())}
V = [("s_all", "전체 논조"), ("s_noprice", "주가 보도 뺀 논조"), ("pos_share", "강한 긍정 비율"), ("neg_share", "강한 부정 비율"),
     ("surprise", "논조 변화(평소 대비)"), ("log_n", "기사 수(log)")]
def icser(x, col, ycol, start="2001-01", end="2021-12"):
    x = x[(x.month >= start) & (x.month <= end)]
    return x.groupby("month").apply(lambda z: z[col].rank().corr(z[ycol].rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False).dropna()
rows = []
for col, lab in V:
    x = u[u[col].notna()]
    c0, c1, c2 = icser(x, col, "ret_r"), icser(x, col, "y"), icser(x, col, "t2_r")
    c1b = icser(x, col, "y", "2011-01")
    rows.append({"변수": lab, "같은달 IC": c0.mean(), "t(같은달)": tt(c0), "다음달 IC": c1.mean(), "t(다음달)": tt(c1),
                 "2달뒤 IC": c2.mean(), "t(2달뒤)": tt(c2), "다음달 IC 2011-21": c1b.mean(), "t(2011-21)": tt(c1b), "달수": len(c1)})
IC = pd.DataFrame(rows); IC.to_csv(f"{P}/sent_ic.csv", index=False, encoding="utf-8-sig"); print(IC.round(3).to_string(index=False))

# 기간별 다음 달 IC (전체 논조)
x = u[u.s_all.notna()]
per = {lab: (lambda s: {"IC": round(s.mean(), 4), "t": round(tt(s), 2)})(icser(x, "s_all", "y", a, b)) for lab, a, b in
       [("2001-10", "2001-01", "2010-12"), ("2011-15", "2011-01", "2015-12"), ("2016-21", "2016-01", "2021-12")]}
R["기간별 다음달 IC(전체 논조)"] = per; print(per)

# 5분위: 전체 논조로 5등분 → 같은 달 / 다음 달 평균 수익률
x = x.copy(); x["q"] = x.groupby("month").s_all.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
Q = x.groupby(["month", "q"])[["ret", "target"]].mean().unstack() * 100
qs = {}
for c, lab in [("ret", "같은 달"), ("target", "다음 달")]:
    d = (Q[c][5] - Q[c][1]).dropna(); qs[lab] = {"Q1": round(Q[c][1].mean(), 3), "Q5": round(Q[c][5].mean(), 3), "Q5-Q1": round(d.mean(), 3), "t": round(tt(d), 2)}
R["5분위(전체 논조)"] = qs; print(qs)
Qm = pd.DataFrame({lab: x.groupby(["month", "q"])[c].mean().unstack().mean() * 100 for c, lab in [("ret", "같은 달"), ("target", "다음 달")]})
Qm.to_csv(f"{P}/sent_quintile.csv", encoding="utf-8-sig")

# FM: 통제 9개 + 전체 논조(순위, 기사 없으면 0.5) + 기사 수
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
u["s_r"] = u.groupby("month").s_all.rank(pct=True).fillna(0.5); u["has"] = u.s_all.notna().astype(float)
u["n_r"] = u.groupby("month").log_n.rank(pct=True)
cols = [c + "_r" for c in ALL] + ["s_r", "n_r", "has"]; B = []
for m, g in u[u.month >= "2002-04"].groupby("month"):
    if len(g) < 30: continue
    X = np.column_stack([np.ones(len(g))] + [g[c] for c in cols]); B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][1:])
B = pd.DataFrame(B, columns=cols); FM = pd.DataFrame({"계수(%p)": B.mean(), "t": B.apply(tt)})
FM.to_csv(f"{P}/sent_fm.csv", encoding="utf-8-sig"); print(FM.round(3).to_string())

# XGBoost 3단계 vs +감성
for c in ALL: u[c + "_m"] = u[c].isna().astype(float)
for c in ["s_all", "s_noprice", "pos_share", "neg_share", "surprise"]: u[c + "_f"] = u[c].fillna(0)
F3 = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]
F4 = F3 + ["s_all_f", "s_noprice_f", "pos_share_f", "neg_share_f", "surprise_f", "log_n", "has"]
def run(F):
    out = []
    for Y in range(2011, 2022):
        tr = u[(u.month <= f"{Y-1}-11") & u.y.notna()]; te = u[(u.month >= f"{Y}-01") & (u.month <= f"{Y}-12")]
        out.append(te[["month", "target"]].assign(score=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda z: z.score.rank().corr(z.target.rank()), include_groups=False)
i3, i4 = run(F3), run(F4); d = i4 - i3
yr = d.groupby(d.index.str[:4]).mean()
R["XGBoost"] = {"3단계 IC": round(i3.mean(), 4), "+감성 IC": round(i4.mean(), 4), "차이": round(d.mean(), 4), "t": round(tt(d), 2), "오른 해": f"{int((yr > 0).sum())}/11"}
print(R["XGBoost"])
json.dump(R, open(f"{P}/sent_summary.json", "w"), ensure_ascii=False, indent=1)
