"""4단계: 뉴스(H3) 쌓기 + H4 설거지 표

입력  out/features.parquet (step1), news_monthly.csv, news_coverage.csv (Colab 노트북 결과)
분석 대상(뉴스 유니버스): 매년 '직전 연도 12월 말 시총 상위 100' 이면서 그해 기사가 실제로 수집된 종목
   → 수집 안 된 종목을 '기사 없음'으로 착각하지 않기 위해
H3 쓸모: XGBoost 누적 학습. 3단계(수급까지) vs 4단계(+화제 논조, +화제 기사 없음 표시)
         · 전체 종목 IC와 · 뉴스 유니버스 안 IC를 둘 다 본다(뉴스는 상위 100에만 있으므로 후자가 본 판정)
H3 방향: Fama-MacBeth(뉴스 유니버스, 통제 9개 + topic_tone) — topic_tone 계수 + & t≥2 → 지지
H4: 호재 종목(화제 기사 있고 논조가 그달 중간값 초과) 안에서 개인 많음−적음 다음 달 수익률 차이
    − 화제 기사 없는 종목에서의 같은 차이 (차이의 차이). − 이고 t≤−2 → 지지
"""
import numpy as np, pandas as pd, matplotlib, sys, os
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, os.path.dirname(__file__))
from step3_xgb_stages import XGBRegressor, XGB, summarize

OUT = "/home/user/fdi-team04-1-/연습장/out"
NEWS = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/news"
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))

# ---------- 데이터 결합 ----------
f = pd.read_parquet(f"{OUT}/features.parquet")
nm = pd.read_csv(f"{NEWS}/news_monthly.csv", dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
cov = pd.read_csv(f"{NEWS}/news_coverage.csv", dtype={"code": str}); cov["code"] = cov.code.str.zfill(6)
cov = cov.melt("code", var_name="year", value_name="n").assign(year=lambda d: d.year.astype(str))
# 유니버스: 직전 연도 12월 말 시총 상위 100
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 100].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
top = top.merge(cov[cov.n > 0], on=["code", "year"], how="left")
top["collected"] = top.n.notna()
cover = top[top.year <= "2021"].groupby("year").collected.mean()
top = top[top.collected][["code", "year"]]
f["year"] = f.month.str[:4]
f = f.merge(top.assign(univ=1), on=["code", "year"], how="left"); f["univ"] = f.univ.fillna(0).astype(int)
f = f.merge(nm, on=["month", "code"], how="left")
for c in ["n_all", "n_topic", "n_disc", "n_contro"]:
    f[c] = f[c].fillna(0)
f["no_topic"] = (f.n_topic == 0).astype(float)
f.loc[f.no_topic == 1, "topic_tone"] = np.nan
print("연도별 상위100 중 기사 수집된 비율:", cover.round(2).to_dict())
print("뉴스 유니버스 종목-월:", f.univ.sum(), " 그중 화제 기사 있음:", ((f.univ == 1) & (f.no_topic == 0)).sum())

# ---------- H3 쓸모: XGBoost 3단계 vs 4단계 ----------
d = f[f.month >= "2001-01"].copy()
for c in ALL + ["topic_tone"]:
    d[c + "_m"] = d[c].isna().astype(float)
    d[c + "_r"] = d.groupby("month")[c].rank(pct=True).fillna(0.5)
d["y"] = d.groupby("month").target.rank(pct=True)
F3 = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]
F4 = F3 + ["topic_tone_r", "no_topic", "univ"]
def expanding(F):
    out = []
    for Y in range(2011, 2022):
        tr = d[(d.month <= f"{Y-1}-11") & d.y.notna()]
        te = d[(d.month >= f"{Y}-01") & (d.month <= f"{Y}-12") & d.target.notna()]
        mdl = XGBRegressor(**XGB).fit(tr[F], tr.y)
        out.append(te[["month", "code", "target", "univ"]].assign(score=mdl.predict(te[F])))
    return pd.concat(out)
def ic(p):
    return p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()) if len(x) >= 10 else np.nan,
                                    include_groups=False).dropna()
p3, p4 = expanding(F3), expanding(F4)
rows = []
for lab, sel in [("전체 종목", lambda p: p), ("뉴스 유니버스(상위100·수집)", lambda p: p[p.univ == 1])]:
    a, b = ic(sel(p3)), ic(sel(p4)); a, b = a.align(b, join="inner")
    r3, _ = summarize("3 수급까지", a); r4, _ = summarize("4 +뉴스(H3)", b, a)
    rows += [{**r3, "평가대상": lab}, {**r4, "평가대상": lab}]
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/stage4_news_ic.csv", index=False, encoding="utf-8-sig")
print(S.round(4).to_string(index=False))

# ---------- H3 방향: Fama-MacBeth (뉴스 유니버스) + 논조 3분위 ----------
u = f[(f.univ == 1) & f.target.notna() & (f.month >= "2002-04")].copy()
for c in ALL + ["topic_tone"]:
    u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(cols, s, e):
    B = []
    for m, g in u[(u.month >= s) & (u.month <= e)].groupby("month"):
        X = np.column_stack([np.ones(len(g))] + [g[c].values for c in cols])
        B.append(np.linalg.lstsq(X, g.target.values * 100, rcond=None)[0][1:])
    B = pd.DataFrame(B, columns=cols)
    return pd.DataFrame({"계수(%p)": B.mean(), "t": B.apply(tt)})
FMc = [c + "_r" for c in ALL] + ["topic_tone_r", "no_topic"]
fmres = pd.concat({"2011~2021": fm(FMc, "2011-01", "2021-11"), "2002~2021": fm(FMc, "2002-04", "2021-11")}, axis=1)
fmres.to_csv(f"{OUT}/fm_news.csv", encoding="utf-8-sig"); print(fmres.round(3).to_string())
w = u[(u.no_topic == 0) & (u.month >= "2011-01")].copy()
w["q"] = w.groupby("month").topic_tone.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False) + 1 if len(s) >= 9 else np.nan)
qm = w.groupby(["month", "q"]).target.mean().unstack() * 100
none = u[(u.no_topic == 1) & (u.month >= "2011-01")].groupby("month").target.mean() * 100
h3 = {"하(부정)": qm[1].mean(), "중": qm[2].mean(), "상(긍정)": qm[3].mean(), "화제 기사 없음": none.mean(),
      "상-하": (qm[3] - qm[1]).mean(), "t": tt(qm[3] - qm[1]), "달수": int((qm[3] - qm[1]).notna().sum())}
print("H3 논조 3분위:", {k: round(v, 3) for k, v in h3.items()})

# ---------- H4 설거지 표 ----------
def h4(df, good_rule="median"):
    res = []
    for m, g in df.groupby("month"):
        news = g[g.no_topic == 0]
        if len(news) < 6: continue
        cut = news.topic_tone.median() if good_rule == "median" else 0.0
        good = news[news.topic_tone > cut]; nonews = g[g.no_topic == 1]
        rmed = g.retail_nb.median()
        def diff(x):
            hi, lo = x[x.retail_nb > rmed], x[x.retail_nb <= rmed]
            return (hi.target.mean() - lo.target.mean()) * 100 if len(hi) >= 2 and len(lo) >= 2 else np.nan, \
                   hi.target.mean() * 100, lo.target.mean() * 100
        dg, gh, gl = diff(good); dn, nh, nl = diff(nonews)
        res.append({"month": m, "호재_개인많음": gh, "호재_개인적음": gl, "없음_개인많음": nh, "없음_개인적음": nl,
                    "호재차이": dg, "없음차이": dn, "DiD": dg - dn})
    return pd.DataFrame(res).set_index("month")
H = []
uu = u[u.month >= "2011-01"].copy()
uu["big"] = uu.groupby("month").me.transform(lambda s: s.rank(pct=True) > 0.5)
for lab, df, rule in [("기본(중간값 초과=호재)", uu, "median"), ("강건: 논조>0=호재", uu, "zero"),
                      ("대형(상위100 중 위 절반)", uu[uu.big], "median"), ("소형(아래 절반)", uu[~uu.big], "median")]:
    r = h4(df, rule)
    H.append({"구분": lab, **r[["호재_개인많음", "호재_개인적음", "없음_개인많음", "없음_개인적음"]].mean().round(3).to_dict(),
              "호재차이": r.호재차이.mean(), "t_호재": tt(r.호재차이), "없음차이": r.없음차이.mean(), "t_없음": tt(r.없음차이),
              "DiD": r.DiD.mean(), "t_DiD": tt(r.DiD), "달수": int(r.DiD.notna().sum())})
    if lab.startswith("기본"): base = r
H = pd.DataFrame(H); H.to_csv(f"{OUT}/h4_table.csv", index=False, encoding="utf-8-sig"); print(H.round(3).to_string(index=False))

# ---------- 그림 ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
x = np.arange(2)
for i, lab in enumerate(["전체 종목", "뉴스 유니버스(상위100·수집)"]):
    s = S[S.평가대상 == lab]
    ax[0].bar(x[i] - 0.18, s.iloc[0].평균IC, 0.36, color="#8c8c8c", label="3 수급까지" if i == 0 else None)
    ax[0].bar(x[i] + 0.18, s.iloc[1].평균IC, 0.36, color="#55a868" if s.iloc[1].판정 == "OK" else "#c44e52",
              label="4 +뉴스" if i == 0 else None)
    ax[0].text(x[i] + 0.18, s.iloc[1].평균IC, f"{s.iloc[1].증가분:+.4f}\n{s.iloc[1]['이긴 해']} {s.iloc[1].판정}", ha="center", va="bottom", fontsize=9)
ax[0].set_xticks(x); ax[0].set_xticklabels(["전체 종목", "상위100(뉴스 수집)"]); ax[0].legend(fontsize=9)
ax[0].set_title("① H3 쓸모: 뉴스를 더하면 IC가 오르나"); ax[0].set_ylabel("평균 IC (2011~2021)")
vals = [h3["하(부정)"], h3["중"], h3["상(긍정)"], h3["화제 기사 없음"]]
ax[1].bar(range(4), vals, color=["#c44e52", "#8c8c8c", "#55a868", "#cccccc"])
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["논조 하", "중", "상", "화제 기사\n없음"])
ax[1].set_title(f"② H3 방향: 화제 논조 3분위 다음 달 수익률\n상−하 {h3['상-하']:+.2f}%p (t {h3['t']:+.1f})"); ax[1].set_ylabel("%")
b = H.iloc[0]
ax[2].bar([0, 1], [b.호재_개인많음, b.호재_개인적음], color=["#c44e52", "#4c72b0"], width=0.6)
ax[2].bar([2.5, 3.5], [b.없음_개인많음, b.없음_개인적음], color=["#c44e52", "#4c72b0"], width=0.6, alpha=0.5)
ax[2].set_xticks([0, 1, 2.5, 3.5]); ax[2].set_xticklabels(["호재\n개인 많음", "호재\n개인 적음", "기사 없음\n개인 많음", "기사 없음\n개인 적음"])
ax[2].set_title(f"③ H4 설거지 표: 차이의 차이 {b.DiD:+.2f}%p (t {b.t_DiD:+.1f})"); ax[2].set_ylabel("다음 달 평균 수익률(%)")
fig.tight_layout(); fig.savefig(f"{OUT}/figI_news_h3h4.png", dpi=130)
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.bar(cover.index, cover.values * 100, color="#4c72b0"); ax.set_ylabel("%"); ax.set_ylim(0, 105)
ax.set_title("그림 J. 해마다 상위 100 종목 중 기사가 실제로 수집된 비율"); ax.set_xticks(range(len(cover))); ax.set_xticklabels(cover.index, rotation=90)
fig.tight_layout(); fig.savefig(f"{OUT}/figJ_news_coverage.png", dpi=130)
