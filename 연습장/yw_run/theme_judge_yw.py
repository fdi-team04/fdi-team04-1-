"""테마별 H3' 스타일 판정 (ESG 분석과 같은 틀) — theme_monthly.csv(영섭 PC 결과) 입력
테마마다: 긍정 있음 / 부정 있음 / 순지표((긍정-부정)/(긍정+부정))
 1) 있음-없음 차이: 같은 달(참고), 다음 달, 2달, 3달
 2) 순지표 IC (같은 달 / 다음 달)
 3) FM: 통제 9개 + 긍정 있음 + 부정 있음 (다음 달)
 4) 기간별(2001-10 / 2011-15 / 2016-21) 다음 달 차이
 5) XGBoost: 3단계(수급까지) vs +테마 변수, 2011-21
판정 기준 (결과 보기 전에 정함, ESG와 같음 + 최근 유지 추가):
  부정: 다음 달 차이 t <= -2  또는  FM 부정 t <= -2
  긍정: 다음 달 차이 t >= 2   또는  FM 긍정 t >= 2
  최근 유지: 통과한 쪽의 2016-21 다음 달 차이가 같은 방향
  → 긍정 또는 부정이 (기준 + 최근 유지) 통과: '지지' (XGBoost IC가 11년 중 6년 이상 오르면 모델 반영 후보)
  → 기준은 통과했지만 2016-21에 방향이 뒤집힘: '보류(최근 소멸)'
  → 둘 다 미통과: '빼기' (긍정·부정 모두 종목-월 300개 미만 또는 2016-21에 30개 미만이면 '표본 부족')
쓰는 법: python theme_judge_yw.py <theme_monthly.csv> [이름표] [--no-xgb]
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/fdi-team04-1-/연습장/news")
from theme_rules import GROUPS
args = [a for a in sys.argv[1:] if not a.startswith("--")]; NOXGB = "--no-xgb" in sys.argv
SRC = args[0]; TAG = args[1] if len(args) > 1 else "full"
O = "/home/user/fdi-team04-1-/연습장/out_yw"; P = f"{O}/theme_judge_{TAG}"; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
nm = pd.read_csv(SRC, dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
f["t2"] = f.groupby("code").target.shift(-1); f["t3"] = f.groupby("code").target.shift(-2)
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left")
u["n_all"] = u.n_all.fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2001-01")].copy()
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5); u[c + "_m"] = u[c].isna().astype(float)
F3 = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]
PER = [("2001–10", "2001-01", "2010-12"), ("2011–15", "2011-01", "2015-12"), ("2016–21", "2016-01", "2021-12")]
def diff(col_flag, ycol):
    d = u.groupby(["month", col_flag])[ycol].mean().unstack() * 100
    return (d[1.0] - d[0.0]).dropna() if 1.0 in d and 0.0 in d else pd.Series(dtype=float)
def icser(col, ycol):
    x = u[u[col].notna()]
    return x.groupby("month").apply(lambda z: z[col].rank().corr(z[ycol].rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False).dropna()
def xgb_ic(F):
    from xgboost import XGBRegressor
    out = []
    for Y in range(2011, 2022):
        tr = u[(u.month <= f"{Y-1}-11") & u.y.notna()]; te = u[(u.month >= f"{Y}-01") & (u.month <= f"{Y}-12")]
        out.append(te[["month", "target"]].assign(score=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda z: z.score.rank().corr(z.target.rank()), include_groups=False)
base = None if NOXGB else xgb_ic(F3)
rows, det = [], []
for g, d in GROUPS.items():
    k = g.replace("·", "_").replace(" ", "_")
    npos = sum(u[c].fillna(0) for c in d["pos"]) if d["pos"] else pd.Series(0, index=u.index)
    nneg = sum(u[c].fillna(0) for c in d["neg"]) if d["neg"] else pd.Series(0, index=u.index)
    u["pos"] = (npos > 0).astype(float); u["neg"] = (nneg > 0).astype(float)
    tot = npos + nneg; u["net"] = np.where(tot > 0, (npos - nneg) / tot.where(tot > 0), np.nan)
    R = {"테마": g, "긍정 종목-월": int(u.pos.sum()), "부정 종목-월": int(u.neg.sum())}
    for side in ["pos", "neg"]:
        if u[side].sum() == 0: continue
        lab = "긍정" if side == "pos" else "부정"
        for col, h in [("ret", "같은 달"), ("target", "다음 달"), ("t2", "2달 뒤"), ("t3", "3달 뒤")]:
            s = diff(side, col); det.append({"테마": g, "신호": lab, "시점": h, "차이(%p)": s.mean(), "t": tt(s)})
        s = diff(side, "target"); R[f"{lab} 다음달 차이"] = s.mean(); R[f"{lab} t"] = tt(s)
        for p, a, b in PER:
            z = s[(s.index >= a) & (s.index <= b)]; R[f"{lab} {p}"] = z.mean(); R[f"{lab} {p} t"] = tt(z)
    c0, c1 = icser("net", "ret_r"), icser("net", "y")
    R["순지표 같은달 IC"] = c0.mean(); R["순지표 같은달 t"] = tt(c0); R["순지표 다음달 IC"] = c1.mean(); R["순지표 다음달 t"] = tt(c1)
    cols = [c + "_r" for c in ALL] + [s for s in ["pos", "neg"] if u[s].sum() > 0]; B = []
    for m, gg in u.groupby("month"):
        if len(gg) < 30: continue
        X = np.column_stack([np.ones(len(gg))] + [gg[c] for c in cols]); B.append(np.linalg.lstsq(X, gg.target * 100, rcond=None)[0][1:])
    B = pd.DataFrame(B, columns=cols)
    for s, lab in [("pos", "긍정"), ("neg", "부정")]:
        if s in B: R[f"FM {lab}"] = B[s].mean(); R[f"FM {lab} t"] = tt(B[s])
    if base is not None:
        u[f"{k}_pos"] = np.log1p(npos); u[f"{k}_neg"] = np.log1p(nneg); u[f"{k}_net"] = u.net.fillna(0)
        i4 = xgb_ic(F3 + [f"{k}_pos", f"{k}_neg", f"{k}_net"]); dd = i4 - base; yr = dd.groupby(dd.index.str[:4]).mean()
        R.update({"XGB 기준 IC": base.mean(), "XGB +테마 IC": i4.mean(), "XGB 차이 t": tt(dd), "XGB 오른 해": int((yr > 0).sum())})
    # 판정
    okp = (R.get("긍정 t", 0) >= 2) or (R.get("FM 긍정 t", 0) >= 2)
    okn = (R.get("부정 t", 0) <= -2) or (R.get("FM 부정 t", 0) <= -2)
    recp = okp and R.get("긍정 2016–21", 0) > 0; recn = okn and R.get("부정 2016–21", 0) < 0
    enough = lambda side: u[side].sum() >= 300 and u[u.month >= "2016-01"][side].sum() >= 30
    R["표본 충분(긍정/부정)"] = f"{enough('pos')}/{enough('neg')}"
    R["판정"] = "지지" if (recp or recn) else ("보류(최근 소멸)" if (okp or okn) else "빼기")
    if R["판정"] == "빼기" and not (enough("pos") or enough("neg")): R["판정"] = "표본 부족"
    R["통과 쪽"] = ",".join([s for s, v in [("긍정", recp), ("부정", recn)] if v]) or "-"
    rows.append(R)
T = pd.DataFrame(rows); T.to_csv(f"{P}/theme_judge.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(det).to_csv(f"{P}/theme_diff_detail.csv", index=False, encoding="utf-8-sig")
show = ["테마", "긍정 종목-월", "부정 종목-월", "표본 충분(긍정/부정)", "긍정 다음달 차이", "긍정 t", "부정 다음달 차이", "부정 t", "FM 긍정 t", "FM 부정 t", "순지표 다음달 IC", "순지표 다음달 t", "판정"]
pd.set_option("display.width", 250); print(T[[c for c in show if c in T]].round(3).to_string(index=False))
if base is not None: print(T[["테마", "XGB 기준 IC", "XGB +테마 IC", "XGB 차이 t", "XGB 오른 해"]].round(4).to_string(index=False))
