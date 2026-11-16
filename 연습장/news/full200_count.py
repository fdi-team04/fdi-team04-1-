"""아영 news_monthly_200 (회사×월 기사 수만 있음)로 '기사 수(관심) 효과'를 전체 200개 회사에서 확인
왜: 파일럿에서 기사 수가 많으면 다음 달 수익률이 낮았다(IC −0.018, t −3.1). 감성 점수는 아직 없지만
기사 수만으로 볼 수 있는 이 효과를 전체 데이터에서 먼저 확인한다.
표본: 그해 '직전 연도 12월 말 시총 상위 200'이면서 그해 기사가 1건 이상인 종목 (수집 대상이 아닌 종목을 '0건'으로 착각하지 않게)
기사 수는 그달 기사 수 → log(1+n). 규모(시총)와 같이 움직이므로 규모를 뺀 '잔차 기사 수'도 본다.
"""
import numpy as np, pandas as pd, json
O = "/home/user/fdi-team04-1-/연습장/out"; P = f"{O}/news_full200"; import os; os.makedirs(P, exist_ok=True)
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "/root/.claude/uploads/03b15712-f400-5188-9b23-fea64cb12856/8515ff1c-news_monthly_200.csv"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
nm = pd.read_csv(SRC, dtype={"code": str})
nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet")
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year", "rk"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm[["code", "month", "n_articles"]], on=["code", "month"], how="left")
u["n"] = u.n_articles.fillna(0)
has = u.groupby(["code", "year"]).n.transform("sum") > 0          # 그해 수집된 종목만
cov = has.groupby(u.year).mean()
u = u[has & u.target.notna() & (u.month >= "2001-01")].copy()
u["logn"] = np.log1p(u.n)
# 규모를 뺀 기사 수: 매월 logn을 log(시총)에 회귀한 잔차
def resid(g):
    X = np.c_[np.ones(len(g)), np.log(g.me)]; b = np.linalg.lstsq(X, g.logn, rcond=None)[0]; return g.logn - X @ b
u["logn_res"] = u.groupby("month", group_keys=False).apply(resid)
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
R = {"표본 종목-월": len(u), "종목": u.code.nunique(), "달": u.month.nunique(), "기사 0건인 달 비율": round((u.n == 0).mean(), 3),
     "연도별 상위200 중 수집 비율(평균)": round(cov.mean(), 3), "기사 수와 log시총 순위상관": round(u.logn.corr(np.log(u.me), method="spearman"), 3)}
rows = []
for col, lab in [("logn", "기사 수(log)"), ("logn_res", "기사 수(규모 제거)")]:
    for per, a, b in [("전체", "2001", "2022"), ("2001–10", "2001", "2011"), ("2011–21", "2011", "2022")]:
        x = u[(u.month >= a) & (u.month < b)]
        c0 = x.groupby("month").apply(lambda z: z[col].rank().corr(z.ret_r.rank()), include_groups=False)
        c1 = x.groupby("month").apply(lambda z: z[col].rank().corr(z.y.rank()), include_groups=False)
        rows.append({"변수": lab, "기간": per, "같은달 IC": c0.mean(), "t(같은달)": tt(c0), "다음달 IC": c1.mean(), "t(다음달)": tt(c1), "달수": len(c1)})
IC = pd.DataFrame(rows); IC.to_csv(f"{P}/count_ic{TAG}.csv", index=False, encoding="utf-8-sig"); print(IC.round(3).to_string(index=False))
# 5분위 (규모 제거 기사 수) 다음 달 수익률
u["q"] = u.groupby("month").logn_res.transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
qm = u.groupby(["month", "q"]).target.mean().unstack() * 100
Q = {f"Q{i}": qm[i].mean() for i in range(1, 6)}; Q["Q5−Q1"] = (qm[5] - qm[1]).mean(); Q["t"] = tt(qm[5] - qm[1])
print("규모 제거 기사 수 5분위 다음 달 수익률(%):", {k: round(v, 3) for k, v in Q.items()})
# FM: 통제 9개 + 기사 수(규모 제거)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
for c in ALL + ["logn_res"]: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
B = []
for m, g in u[u.month >= "2002-04"].groupby("month"):
    X = np.column_stack([np.ones(len(g))] + [g[c + "_r"] for c in ALL + ["logn_res"]])
    B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][1:])
B = pd.DataFrame(B, columns=ALL + ["logn_res"]); FM = pd.DataFrame({"계수(%p)": B.mean(), "t": B.apply(tt)})
FM.to_csv(f"{P}/count_fm{TAG}.csv", encoding="utf-8-sig"); print(FM.round(3).to_string())
R["5분위"] = {k: round(v, 3) for k, v in Q.items()}
json.dump(R, open(f"{P}/count_sample{TAG}.json", "w"), ensure_ascii=False, indent=1); print(R)
