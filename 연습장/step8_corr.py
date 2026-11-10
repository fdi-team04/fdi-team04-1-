"""그림 H: 변수끼리 겹침(매달 순위상관의 평균, 2011~2021)"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
f = pd.read_parquet("out/features.parquet"); f = f[f.month >= "2011-01"]
c = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
nm = ["베타", "규모", "가치", "단기반전", "모멘텀", "고유변동성", "주식발행", "개인순매수", "외국인순매수"]
m = f.groupby("month")[c].apply(lambda x: x.rank().corr()).groupby(level=1).mean().loc[c, c]
m.to_csv("out/var_corr.csv", encoding="utf-8-sig")
fig, ax = plt.subplots(figsize=(8, 6.5)); im = ax.imshow(m, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
ax.set_xticks(range(9)); ax.set_xticklabels(nm, rotation=45, ha="right"); ax.set_yticks(range(9)); ax.set_yticklabels(nm)
for i in range(9):
    for k in range(9):
        ax.text(k, i, f"{m.iloc[i, k]:.2f}", ha="center", va="center", fontsize=8, color="w" if abs(m.iloc[i, k]) > 0.4 else "k")
fig.colorbar(im, shrink=.8); ax.set_title("그림 H. 변수끼리 얼마나 겹치나 (매달 순위상관 평균, 2011~2021)")
fig.tight_layout(); fig.savefig("out/figH_corr.png", dpi=130)
