"""ESG v1.2 결과 그림(f4와 같은 모양) + 기간별 표"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
FD = "/tmp/claude-0/-home-user-fdi-team04-1-/03b15712-f400-5188-9b23-fea64cb12856/scratchpad/fonts"
for f in ("plex400.ttf", "plex600.ttf"): fm.fontManager.addfont(f"{FD}/{f}")
plt.rcParams.update({"font.family": "IBM Plex Sans KR", "axes.unicode_minus": False, "font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
O = "/home/user/fdi-team04-1-/연습장/out_yw"; P = f"{O}/news_esg_v12"
D = pd.read_csv(f"{P}/esg_diff.csv"); H = ["같은 달", "다음 달", "2달 뒤", "3달 뒤"]
fig, ax = plt.subplots(figsize=(13, 4.8)); x = np.arange(4)
for i, (lab, c, nm) in enumerate([("부정 ESG 있음−없음", "#D9822B", "부정 ESG 기사가 있는 종목"), ("긍정 ESG 있음−없음", "#1E4E8C", "긍정 ESG 기사가 있는 종목")]):
    d = D[D.비교 == lab].set_index("시점").loc[H]; se = d["차이(%p)"] / d.t
    ax.bar(x + (i - 0.5) * 0.36, d["차이(%p)"], 0.34, color=c, label=nm, yerr=2 * se, capsize=4, error_kw=dict(ecolor="#333", lw=1.2))
ax.axhline(0, color="#8C877D", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(H); ax.set_ylabel("수익률 차이 (%p)"); ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{O}/f4_esg_v12.png", dpi=140); fig.savefig("/home/user/fdi-team04-1-/연습장/해설서/img_final/f4_esg_v12.png", dpi=140)
# 기간별
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x)))
f = pd.read_parquet(f"{O}/features.parquet")
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
m = pd.read_csv("/home/user/fdi-team04-1-/연습장/out/news_esg_v12_input.csv", dtype={"code": str}); m["code"] = m.code.str.zfill(6)
u = f.merge(top, on=["code", "year"]).merge(m, on=["month", "code"], how="left"); u["n_all"] = u.n_all.fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2001-01")]
rows = []
for col, lab in [("n_esg_neg", "부정"), ("n_esg_pos", "긍정")]:
    u["fl"] = (u[col].fillna(0) > 0).astype(int)
    d = u.groupby(["month", "fl"]).target.mean().unstack() * 100; d = (d[1] - d[0]).dropna()
    for p, a, b in [("2001–10", "2001", "2010-12"), ("2011–15", "2011", "2015-12"), ("2016–21", "2016", "2021-12"), ("전체", "2001", "2021-12")]:
        z = d[(d.index >= a) & (d.index <= b)]; rows.append({"ESG": lab, "기간": p, "다음달 차이(%p)": z.mean(), "t": tt(z)})
R = pd.DataFrame(rows); R.to_csv(f"{P}/esg_by_period.csv", index=False, encoding="utf-8-sig"); print(R.round(2).to_string(index=False))
