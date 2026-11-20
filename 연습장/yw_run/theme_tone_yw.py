"""실적·애널리스트 테마 기사 논조 → 다음 달 (theme_tone_monthly.csv, 영섭 PC 결과)
질문: 테마 기사가 있는 종목끼리 비교할 때, 논조(감성 점수)가 좋을수록 다음 달 수익률이 높은가?
판정 기준 (결과 보기 전에 정함, 11/20):
  ① 논조 다음 달 IC (그 테마 기사가 있는 종목끼리) t >= 2  또는  ② 통제 9개 FM에서 논조 계수 t >= 2
  그리고 ③ 2016-21 IC 또는 FM 계수가 + (뒤집히지 않음)
  → ①또는② + ③: '논조도 지지' / ①②만: '보류(최근 소멸)' / 둘 다 아님: '논조로는 안 됨(있음/없음 신호만)'
"""
import sys, os, numpy as np, pandas as pd
SRC = sys.argv[1]; O = "/home/user/fdi-team04-1-/연습장/out_yw"; P = f"{O}/theme_tone"; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
nm = pd.read_csv(SRC, dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
for c in [c for c in nm if c.startswith(("tone_", "n_"))]: nm[c] = pd.to_numeric(nm[c], errors="coerce")
f = pd.read_parquet(f"{O}/features.parquet")
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left"); u["n_all"] = u.n_all.fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna()].copy()
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
def ic(col, ycol, d):
    x = d[d[col].notna()]
    if x.empty: return pd.Series(dtype=float)
    r = x.groupby("month").apply(lambda z: z[col].rank().corr(z[ycol].rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False)
    return pd.to_numeric(pd.Series(r), errors="coerce").dropna() if isinstance(r, pd.Series) else pd.Series(dtype=float)
def fm(d, col):
    d = d.copy(); d["tr"] = d.groupby("month")[col].rank(pct=True).fillna(0.5); d["has"] = d[col].notna().astype(float); B = []
    for m, g in d.groupby("month"):
        if len(g) < 30 or g.has.sum() < 3: continue
        X = np.column_stack([np.ones(len(g))] + [g[c + "_r"] for c in ALL] + [g.has, g.tr]); B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][-1])
    return pd.Series(B)
rows = []
for col, lab in [("tone_실적", "실적 기사 전체 논조"), ("tone_실적_pos", "실적 호재 기사 논조"), ("tone_애널", "애널 기사 전체 논조"), ("tone_애널_pos", "목표가 상향 기사 논조")]:
    if col not in u: continue
    rec = u[u.month >= "2016-01"]
    c0, c1, c1r = ic(col, "ret_r", u), ic(col, "y", u), ic(col, "y", rec); b, br = fm(u, col), fm(rec, col)
    ok = (tt(c1) >= 2) or (tt(b) >= 2); recent = (c1r.mean() > 0) or (br.mean() > 0)
    rows.append({"신호": lab, "종목-월": int(u[col].notna().sum()), "같은달 IC": c0.mean(), "t같은달": tt(c0),
                 "다음달 IC": c1.mean(), "t다음달": tt(c1), "다음달 IC 2016-21": c1r.mean(), "t최근": tt(c1r),
                 "FM 계수(%p)": b.mean(), "tFM": tt(b), "FM 2016-21": br.mean(), "tFM최근": tt(br),
                 "판정": "논조도 지지" if ok and recent else ("보류(최근 소멸)" if ok else "논조로는 안 됨")})
T = pd.DataFrame(rows); T.to_csv(f"{P}/theme_tone.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(T.round(3).to_string(index=False))
