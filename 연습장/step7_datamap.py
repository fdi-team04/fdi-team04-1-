"""그림 0: 데이터 지도 — 종목 수, 변수 채움 비율, 극단 수익률·퇴장, 외국인 0 비율"""
import pandas as pd, numpy as np, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data"; OUT = "out"
f = pd.read_parquet(f"{OUT}/features.parquet")
j = pd.read_csv(f"{D}/joined_investor.csv", dtype={"Stock_Code": str}); j["month"] = j.Month.str[:7]
r = j[j.adjusted_monthly_return.notna()]
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
n = f.groupby("month").size(); ax[0, 0].plot(pd.to_datetime(n.index), n, color="#4c72b0")
ax[0, 0].axvspan(pd.Timestamp("2000-01"), pd.Timestamp("2000-12"), color="#ddd"); ax[0, 0].text(pd.Timestamp("2000-02"), n.min() + 5, "warmup", fontsize=9)
ax[0, 0].axvspan(pd.Timestamp("2011-01"), pd.Timestamp("2021-12"), color="#e8f0e0"); ax[0, 0].text(pd.Timestamp("2014-01"), n.min() + 5, "검증 2011~2021", fontsize=9)
ax[0, 0].set_title("① 매달 수익률이 있는 종목 수 (636 ~ 752개)"); ax[0, 0].set_ylabel("종목 수")
fill = f.groupby("month")[["bm", "beta", "mom", "ivol"]].apply(lambda x: x.notna().mean() * 100)
for c, lab, col in [("bm", "가치(1/PBR)", "#c44e52"), ("beta", "베타", "#4c72b0"), ("mom", "모멘텀", "#55a868"), ("ivol", "고유변동성", "#dd8452")]:
    ax[0, 1].plot(pd.to_datetime(fill.index), fill[c], label=lab, color=col)
ax[0, 1].annotate("PBR 파일 2002-04부터", (pd.Timestamp("2002-04"), 5), xytext=(pd.Timestamp("2004-06"), 20), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax[0, 1].annotate("2008-08~11 PBR 통째로 없음", (pd.Timestamp("2008-09"), 2), xytext=(pd.Timestamp("2010-01"), 35), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax[0, 1].set_title("② 변수가 채워진 비율 (%)"); ax[0, 1].legend(fontsize=9, loc="lower right"); ax[0, 1].set_ylim(-3, 103)
yr = r.month.str[:4]
ext = r[r.adjusted_monthly_return > 1].groupby(yr).size()
last = r.groupby("Stock_Code").month.max(); ex = last[last < "2021-12"].str[:4].value_counts().sort_index()
x = np.arange(len(ext)); ax[1, 0].bar(x - 0.2, ext.values, 0.4, label="월 수익률 +100% 초과", color="#dd8452")
ax[1, 0].bar(x + 0.2, ex.reindex(ext.index).fillna(0).values, 0.4, label="그해 사라진(퇴장) 종목", color="#8c8c8c")
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(ext.index, rotation=90, fontsize=8); ax[1, 0].legend(fontsize=9)
ax[1, 0].set_title("③ 극단 수익률(총 460건)과 퇴장 종목(총 291개), 연도별")
fz = r.groupby(yr).apply(lambda x: (x.foreign == 0).mean() * 100, include_groups=False)
rz = r.groupby(yr).apply(lambda x: (x.adjusted_monthly_return == 0).mean() * 100, include_groups=False)
ax[1, 1].bar(fz.index, fz.values, color="#4c72b0", label="외국인 순매수 = 0 인 칸")
ax[1, 1].plot(rz.index, rz.values, color="#c44e52", marker="o", ms=3, label="월 수익률이 정확히 0인 칸")
ax[1, 1].set_xticks(range(len(fz))); ax[1, 1].set_xticklabels(fz.index, rotation=90, fontsize=8)
ax[1, 1].set_title("④ 외국인 거래가 0인 칸 비율(%) — 2000년대 초 소형주"); ax[1, 1].legend(fontsize=9)
fig.suptitle("그림 0. 데이터 지도: 무엇이 언제 비어 있고, 어디가 튀는가", fontsize=14)
fig.tight_layout(); fig.savefig(f"{OUT}/fig0_datamap.png", dpi=130)
