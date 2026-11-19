"""H5 논란형 빠른 확인 (11/18 피드백: 한 명이 먼저 찍어 보기)
신호는 모두 t월 말까지의 정보만 사용 (다음 달 기사로 고르지 않음)
  급증: t월 부정 ESG 기사 >= 5 이고, 직전 12개월(t-12~t-1) 평균의 3배 이상
  지속 논란: 급증 & t-1월 부정 기사 >= 3   /  신규 논란: 급증 & t-1월 < 3
예측 대상: t+1월(다음 달), t+2, t+3 수익률. 같은 달(t) 수익률은 참고로만.
판단 기준(미리 정함): 지속 논란 사건 >= 30, 그리고 다음 달 또는 1~3개월 누적 차이 t <= -2 → 진행
입력: out/news_esg_v12_input.csv (규칙 v1.2), out_yw/features.parquet
"""
import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f_ in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f_}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
B = "/home/user/fdi-team04-1-/연습장"; O = f"{B}/out_yw"; P = f"{O}/h5_controversy"; import os; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
R = {}
# 뉴스: 회사별 연속 월 패널로 채운 뒤 과거 평균 계산
nm = pd.read_csv(f"{B}/out/news_esg_v12_input.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
f["t2"] = f.groupby("code").target.shift(-1); f["t3"] = f.groupby("code").target.shift(-2)
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(nm[["month", "code", "n_all", "n_esg_neg"]], on=["month", "code"], how="left")
u[["n_all", "n_esg_neg"]] = u[["n_all", "n_esg_neg"]].fillna(0)
u = u.sort_values(["code", "month"])
g = u.groupby("code").n_esg_neg
u["neg_l1"] = g.shift(1)
u["neg_avg12"] = g.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())
u["surge"] = (u.n_esg_neg >= 5) & (u.n_esg_neg >= 3 * u.neg_avg12) & u.neg_avg12.notna()
u["persist"] = u.surge & (u.neg_l1 >= 3)
u["new"] = u.surge & (u.neg_l1 < 3)
u = u.merge(top, on=["code", "year"])
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2002-01")].copy()
R["표본 종목-월"] = len(u)
R["사건 수"] = {"급증": int(u.surge.sum()), "지속 논란": int(u.persist.sum()), "신규 논란": int(u.new.sum()),
              "지속 논란 종목 수": int(u[u.persist].code.nunique())}
R["연도별 지속 논란"] = u[u.persist].groupby("year").size().to_dict()
print(json.dumps(R, ensure_ascii=False, indent=1))
# 1. 있음−없음 차이 (같은 달 안에서): 같은 달(참고) / 다음 달 / 2달 / 3달 / 1~3개월 누적
u["cum3"] = (1 + u.target) * (1 + u.t2) * (1 + u.t3) - 1
rows = []
for flag, lab in [("persist", "지속 논란"), ("new", "신규 논란"), ("surge", "급증 전체")]:
    for col, h in [("ret", "같은 달(참고)"), ("target", "다음 달"), ("t2", "2달 뒤"), ("t3", "3달 뒤"), ("cum3", "1~3개월 누적")]:
        x = u[u[col].notna()]
        d = x.groupby(["month", flag])[col].mean().unstack() * 100
        if True not in d: continue
        dd = (d[True] - d[False]).dropna(); rows.append({"신호": lab, "시점": h, "차이(%p)": dd.mean(), "t": tt(dd), "사건 있는 달 수": len(dd)})
D = pd.DataFrame(rows); D.to_csv(f"{P}/h5_diff.csv", index=False, encoding="utf-8-sig"); print(D.round(2).to_string(index=False))
# 2. 기간별 (지속 논란, 다음 달 / 1~3개월 누적)
rows = []
for col, h in [("target", "다음 달"), ("cum3", "1~3개월 누적")]:
    x = u[u[col].notna()]; d = x.groupby(["month", "persist"])[col].mean().unstack() * 100; dd = (d[True] - d[False]).dropna()
    for p, a, b in [("2002–10", "2002", "2010-12"), ("2011–15", "2011", "2015-12"), ("2016–21", "2016", "2021-12")]:
        z = dd[(dd.index >= a) & (dd.index <= b)]; rows.append({"시점": h, "기간": p, "차이(%p)": z.mean(), "t": tt(z), "달 수": len(z)})
PP = pd.DataFrame(rows); PP.to_csv(f"{P}/h5_by_period.csv", index=False, encoding="utf-8-sig"); print(PP.round(2).to_string(index=False))
# 3. FM: 통제 9개 + 지속 + 신규 (다음 달)
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
cols = [c + "_r" for c in ALL] + ["persist", "new"]; Bt = []
for m, gg in u.groupby("month"):
    if len(gg) < 30: continue
    X = np.column_stack([np.ones(len(gg))] + [gg[c].astype(float) for c in cols]); Bt.append(np.linalg.lstsq(X, gg.target * 100, rcond=None)[0][1:])
Bt = pd.DataFrame(Bt, columns=cols); FM = pd.DataFrame({"계수(%p)": Bt.mean(), "t": Bt.apply(tt)}); FM.to_csv(f"{P}/h5_fm.csv", encoding="utf-8-sig")
print(FM.loc[["persist", "new"]].round(3).to_string())
# 4. 사건 전후 그림: 사건 달(0) 기준 -3~+6개월, 같은 달 표본 평균 대비 초과수익
u["exc"] = u.ret - u.groupby("month").ret.transform("mean")
full = f.merge(top, on=["code", "year"])[["code", "month", "ret"]].copy()
full["exc"] = full.ret - full.groupby("month").ret.transform("mean")
full = full.sort_values(["code", "month"]); full["i"] = full.groupby("code").cumcount()
idx = full.set_index(["code", "month"]).i
fig, ax = plt.subplots(figsize=(11, 4.6)); K = range(-3, 7)
for flag, lab, c in [("persist", "지속 논란", "#C2611A"), ("new", "신규 논란", "#1E4E8C")]:
    ev = u[u[flag]][["code", "month"]]; ev = ev[ev.set_index(["code", "month"]).index.isin(idx.index)]
    ev["i"] = idx.loc[list(zip(ev.code, ev.month))].values
    lk = full.set_index(["code", "i"]).exc
    mean = []
    for k in K:
        keys = list(zip(ev.code, ev.i + k)); v = lk.reindex(keys).dropna() * 100; mean.append(v.mean())
    ax.plot(list(K), np.cumsum(np.nan_to_num(mean)), marker="o", color=c, label=f"{lab} ({len(ev)}건)")
ax.axvline(0, color="#999", ls="--", lw=0.8); ax.axvline(0.5, color="#C2611A", ls=":", lw=1); ax.axhline(0, color="#555", lw=0.8)
ax.text(0.6, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.5, "← 이달 말 신호 확정 | 예측 구간 →", fontsize=10, color="#C2611A")
ax.set_xlabel("사건 달(0) 기준 개월"); ax.set_ylabel("누적 초과수익 (%p, 상위200 평균 대비)")
ax.set_title("부정 기사 급증 전후 누적 초과수익 (0월까지는 이미 지난 수익, 1월부터가 예측 대상)", loc="left", fontsize=12)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{P}/h5_event.png", dpi=140)
json.dump(R, open(f"{P}/h5_summary.json", "w"), ensure_ascii=False, indent=1)
