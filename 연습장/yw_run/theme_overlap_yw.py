"""실적 호재 vs 애널리스트 목표가 상향: 겹치는지, 같이 넣어도 각자 남는지 (다음 달, 통제 9개 FM)"""
import numpy as np, pandas as pd
O = "/home/user/fdi-team04-1-/연습장/out_yw"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
nm = pd.read_csv("/home/user/fdi-team04-1-/연습장/out/theme_monthly_input.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet")
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left"); u["n_all"] = u.n_all.fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna()].copy()
u["E"] = (u.실적_호재.fillna(0) > 0).astype(float); u["A"] = (u.목표가_상향.fillna(0) > 0).astype(float)
print("실적 호재 있는 종목-월", int(u.E.sum()), "| 목표가 상향", int(u.A.sum()), "| 둘 다", int((u.E * u.A).sum()))
print("실적 호재 달 중 목표가 상향도 있는 비율", round((u.E * u.A).sum() / u.E.sum(), 3), "| 목표가 상향 달 중 실적 호재도 있는 비율", round((u.E * u.A).sum() / u.A.sum(), 3))
print("종목-월 상관(있음 여부)", round(u[["E", "A"]].corr().iloc[0, 1], 3))
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(cols, d):
    B = []
    for m, g in d.groupby("month"):
        if len(g) < 30: continue
        X = np.column_stack([np.ones(len(g))] + [g[c + "_r"] for c in ALL] + [g[c] for c in cols]); B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][-len(cols):])
    B = pd.DataFrame(B, columns=cols); return {c: f"{B[c].mean():+.2f}%p (t {tt(B[c]):.1f})" for c in cols}
rows = []
for nm_, d in [("2001–21", u), ("2016–21", u[u.month >= "2016-01"])]:
    rows += [{"기간": nm_, "모형": "실적만", **fm(["E"], d)}, {"기간": nm_, "모형": "애널만", **fm(["A"], d)}, {"기간": nm_, "모형": "둘 다", **fm(["E", "A"], d)}]
R = pd.DataFrame(rows).rename(columns={"E": "실적 호재", "A": "목표가 상향"}).fillna("-")
R.to_csv(f"{O}/theme_judge_full/overlap_E_A.csv", index=False, encoding="utf-8-sig"); print(R.to_string(index=False))
