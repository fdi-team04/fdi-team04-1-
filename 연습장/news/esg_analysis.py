"""ESG 뉴스 → 다음 달 수익률 (새 H3·H4). 코랩 결과 news_monthly_200_scored.csv 가 오면 이 파일 하나로 끝까지 돌림

쓰는 법:  python esg_analysis.py <news_monthly.csv> <결과폴더 이름표>
입력 칸: month, code, n_all, n_esg_neg, n_esg_pos, esg_net, (선택) esg_neg_tone, esg_pos_tone
표본: 해마다 직전 12월 말 시총 상위 200 × 그해 기사 1건 이상 (수집 안 된 종목을 '뉴스 없음'으로 착각하지 않게)

새 가설 (11/16 교수님 피드백: 뉴스를 ESG 틀로 좁힘)
 H3': ESG 부정 사건이 있는 종목은 다음 달(그리고 그 뒤 2~3달) 수익률이 낮다 / ESG 순지표가 높을수록 다음 달 수익률이 높다
      → 판정: 부정 있음−없음 차이가 − 이고 t≤−2, 또는 FM(통제 9개)에서 ESG 변수 t≥2(방향 맞게), 그리고 XGBoost 4단계 IC 증가
 H4': (a) 덮기: 부정 사건이 있는 달에 홍보(긍정) 기사도 있으면 하락이 덜한가?
      (b) 설거지: 부정 사건 종목을 개인이 많이 산 경우 다음 달 더 약한가? (부정 사건 없는 종목과의 차이의 차이)
왜 2~3달 뒤까지 보나: 교수님 강의(버냉키 2005) — 시장이 바로 반영 못 한 정보만 예측력이 있음.
   사건·사고는 피해 규모를 바로 알기 어려워 늦게 반영될 수 있다는 것이 이 가설의 논리.
"""
import sys, os, json, numpy as np, pandas as pd
from xgboost import XGBRegressor
SRC = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "full"
O = "/home/user/fdi-team04-1-/연습장/out"; P = f"{O}/news_esg_{TAG}"; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
R = {}

nm = pd.read_csv(SRC, dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
# 2·3달 뒤 수익률 (지연 반응 확인용): target = m+1 달 수익률
f["t2"] = f.groupby("code").target.shift(-1); f["t3"] = f.groupby("code").target.shift(-2)
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left")
for c in ["n_all", "n_esg_neg", "n_esg_pos"]: u[c] = u[c].fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2001-01")].copy()
u["neg"] = (u.n_esg_neg > 0).astype(float); u["pos"] = (u.n_esg_pos > 0).astype(float)
u["y"] = u.groupby("month").target.rank(pct=True); u["ret_r"] = u.groupby("month").ret.rank(pct=True)
R["표본"] = {"종목-월": len(u), "종목": int(u.code.nunique()), "달": int(u.month.nunique()),
            "부정 ESG 있는 종목-월 비율": round(u.neg.mean(), 4), "긍정 ESG 있는 비율": round(u.pos.mean(), 4)}

# 1. 있음−없음 차이 (같은 달 안에서 비교): 같은 달 / 다음 달 / 2달 뒤 / 3달 뒤
rows = []
for flag, lab in [("neg", "부정 ESG 있음−없음"), ("pos", "긍정 ESG 있음−없음")]:
    for col, h in [("ret", "같은 달"), ("target", "다음 달"), ("t2", "2달 뒤"), ("t3", "3달 뒤")]:
        d = u.groupby(["month", flag])[col].mean().unstack() * 100
        if 1 not in d or 0 not in d: continue
        dd = (d[1] - d[0]).dropna(); rows.append({"비교": lab, "시점": h, "차이(%p)": dd.mean(), "t": tt(dd), "달수": len(dd)})
D = pd.DataFrame(rows); D.to_csv(f"{P}/esg_diff.csv", index=False, encoding="utf-8-sig"); print(D.round(3).to_string(index=False))

# 2. IC (순위상관)
rows = []
for col, lab in [("esg_net", "ESG 순지표(ESG 기사 있는 칸)"), ("neg", "부정 있음"), ("pos", "긍정 있음")] + \
                ([("esg_neg_tone", "부정 기사 논조")] if "esg_neg_tone" in u and u.esg_neg_tone.notna().sum() > 100 else []):
    x = u[u[col].notna()]
    def ic(ycol):
        return x.groupby("month").apply(lambda z: z[col].rank().corr(z[ycol].rank()) if z[col].nunique() > 1 and len(z) >= 8 else np.nan, include_groups=False).dropna()
    c0, c1 = ic("ret_r"), ic("y")
    rows.append({"변수": lab, "같은달 IC": c0.mean(), "t(같은달)": tt(c0), "다음달 IC": c1.mean(), "t(다음달)": tt(c1), "달수": len(c1)})
IC = pd.DataFrame(rows); IC.to_csv(f"{P}/esg_ic.csv", index=False, encoding="utf-8-sig"); print(IC.round(3).to_string(index=False))

# 3. Fama-MacBeth: 통제 9개 + 부정 있음 + 긍정 있음
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
B = []
cols = [c + "_r" for c in ALL] + ["neg", "pos"]
for m, g in u[u.month >= "2002-04"].groupby("month"):
    if len(g) < 30: continue
    X = np.column_stack([np.ones(len(g))] + [g[c] for c in cols]); B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][1:])
B = pd.DataFrame(B, columns=cols); FM = pd.DataFrame({"계수(%p)": B.mean(), "t": B.apply(tt)})
FM.to_csv(f"{P}/esg_fm.csv", encoding="utf-8-sig"); print(FM.round(3).to_string())

# 4. XGBoost 3단계(수급까지) vs 4단계(+ESG) — 같은 누적 학습, 같은 표본
for c in ALL: u[c + "_m"] = u[c].isna().astype(float)
u["esg_net_f"] = u.esg_net.fillna(0); u["log_neg"] = np.log1p(u.n_esg_neg); u["log_pos"] = np.log1p(u.n_esg_pos)
F3 = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]
F4 = F3 + ["neg", "pos", "log_neg", "log_pos", "esg_net_f"]
def run(F):
    out = []
    for Y in range(2011, 2022):
        tr = u[(u.month <= f"{Y-1}-11") & u.y.notna()]; te = u[(u.month >= f"{Y}-01") & (u.month <= f"{Y}-12")]
        out.append(te[["month", "target"]].assign(score=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out); return p.groupby("month").apply(lambda z: z.score.rank().corr(z.target.rank()), include_groups=False)
i3, i4 = run(F3), run(F4); d = i4 - i3
yr = (i4.groupby(i4.index.str[:4]).mean() > i3.groupby(i3.index.str[:4]).mean()).sum()
X = {"3단계 IC": i3.mean(), "4단계(+ESG) IC": i4.mean(), "차이": d.mean(), "차이 t": tt(d), "4단계가 나은 해(/11)": int(yr)}
print({k: round(v, 4) for k, v in X.items()}); R["XGBoost"] = X

# 5. H4' (a) 덮기 (b) 설거지
w = u[u.neg == 1]
b = w.groupby(["month", "pos"]).target.mean().unstack() * 100
cover = (b[1] - b[0]).dropna() if 1 in b and 0 in b else pd.Series(dtype=float)
def retail_gap(df):
    r = []
    for m, g in df.groupby("month"):
        if len(g) < 4: continue
        med = g.retail_nb.median(); hi, lo = g[g.retail_nb > med], g[g.retail_nb <= med]
        if len(hi) and len(lo): r.append((m, (hi.target.mean() - lo.target.mean()) * 100))
    return pd.Series(dict(r))
gn = retail_gap(u[u.neg == 1]); g0 = retail_gap(u[(u.neg == 0) & (u.pos == 0)]); did = (gn - g0).dropna()
H4 = {"(a) 부정 있는 종목 중 홍보도 있음−없음(%p)": cover.mean(), "(a) t": tt(cover), "(a) 달수": len(cover),
      "(b) 부정 종목 안 개인 많음−적음(%p)": gn.mean(), "(b) ESG 없는 종목 안 같은 차이": g0.mean(), "(b) 차이의 차이": did.mean(), "(b) t": tt(did), "(b) 달수": len(did)}
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in H4.items()}); R["H4'"] = H4
json.dump(R, open(f"{P}/esg_summary.json", "w"), ensure_ascii=False, indent=1, default=float)
