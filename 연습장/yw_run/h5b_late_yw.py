"""H5-B 월말 진행 중 논란 (11/19 정의, 결과 보기 전에 정함)
신호(모두 t월 말까지의 정보): 급증(H5와 같음: 부정 ESG >= 5건, 직전 12개월 평균의 3배 이상) 이면서
  - 월말 진행: 그달 마지막 7일(마감 전)에도 부정 기사 1건 이상  → 이달 말에도 논란이 끝나지 않음
  - 월초 종료: 급증했지만 마지막 7일엔 부정 기사 없음 (비교용)
판단 기준: 월말 진행 사건 >= 30, 그리고 다음 달 또는 1~3개월 누적 차이 t <= -2
입력: out/esg_late_monthly_input.csv (영섭 PC, 규칙 v1.2), out_yw/features.parquet
"""
import numpy as np, pandas as pd, json, os
B = "/home/user/fdi-team04-1-/연습장"; O = f"{B}/out_yw"; P = f"{O}/h5b_late"; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
nm = pd.read_csv(f"{B}/out/esg_late_monthly_input.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
f["t2"] = f.groupby("code").target.shift(-1); f["t3"] = f.groupby("code").target.shift(-2)
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(nm, on=["month", "code"], how="left")
for c in ["n_all", "n_neg", "n_neg_late"]: u[c] = u[c].fillna(0)
u = u.sort_values(["code", "month"])
u["avg12"] = u.groupby("code").n_neg.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
u["surge"] = (u.n_neg >= 5) & (u.n_neg >= 3 * u.avg12) & u.avg12.notna()
u["late"] = u.surge & (u.n_neg_late >= 1); u["early"] = u.surge & (u.n_neg_late == 0)
u = u.merge(top, on=["code", "year"])
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2002-01")].copy()
u["cum3"] = (1 + u.target) * (1 + u.t2) * (1 + u.t3) - 1
R = {"사건 수": {"급증": int(u.surge.sum()), "월말 진행": int(u.late.sum()), "월초 종료": int(u.early.sum())}}
rows = []
for flag, lab in [("late", "월말 진행"), ("early", "월초 종료")]:
    for col, h in [("ret", "같은 달(참고)"), ("target", "다음 달"), ("t2", "2달 뒤"), ("t3", "3달 뒤"), ("cum3", "1~3개월 누적")]:
        x = u[u[col].notna()]; d = x.groupby(["month", flag])[col].mean().unstack() * 100
        dd = (d[True] - d[False]).dropna(); rows.append({"신호": lab, "시점": h, "차이(%p)": dd.mean(), "t": tt(dd), "달 수": len(dd)})
D = pd.DataFrame(rows); D.to_csv(f"{P}/h5b_diff.csv", index=False, encoding="utf-8-sig")
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
cols = [c + "_r" for c in ALL] + ["late", "early"]; Bt = []
for m, g in u.groupby("month"):
    if len(g) < 30: continue
    X = np.column_stack([np.ones(len(g))] + [g[c].astype(float) for c in cols]); Bt.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][1:])
Bt = pd.DataFrame(Bt, columns=cols); FM = pd.DataFrame({"계수(%p)": Bt.mean(), "t": Bt.apply(tt)}).loc[["late", "early"]]
FM.to_csv(f"{P}/h5b_fm.csv", encoding="utf-8-sig")
nx = D[(D.신호 == "월말 진행") & (D.시점 == "다음 달")].iloc[0]; cm = D[(D.신호 == "월말 진행") & (D.시점 == "1~3개월 누적")].iloc[0]
R["판정"] = "진행" if (R["사건 수"]["월말 진행"] >= 30 and (nx.t <= -2 or cm.t <= -2)) else "종료"
json.dump(R, open(f"{P}/h5b_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False)); print(D.round(2).to_string(index=False)); print(FM.round(3).to_string())
