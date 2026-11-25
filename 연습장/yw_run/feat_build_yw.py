"""후보 정형 피처 14개 만들기 (기록/피처판정기준_1125.md). 모든 피처는 그달 말까지의 정보만 사용."""
import numpy as np, pandas as pd
y = pd.read_csv("/home/user/data/yewon/x/final_structured_data_raw.csv", dtype={"Stock_Code": str})
y["month"] = y.Month.str[:7]; y["code"] = y.Stock_Code.str.zfill(6); y = y.sort_values(["code", "month"]).reset_index(drop=True)
r = y.adjusted_monthly_return; g = y.groupby("code")
lr = np.log1p(r.clip(lower=-0.99))
def roll(s, w, fn, minp):
    return s.groupby(y.code).transform(lambda x: getattr(x.rolling(w, min_periods=minp), fn)())
cum = lambda a, b: (lr.groupby(y.code).transform(lambda x: x.shift(a).rolling(b - a + 1, min_periods=int((b - a + 1) * 0.8)).sum()))
F = pd.DataFrame({"month": y.month, "code": y.code})
F["mom6"] = cum(1, 5)                     # t-5..t-1 (6-1)
F["ltrev"] = cum(12, 35)                  # t-35..t-12
idx = lr.fillna(0).groupby(y.code).cumsum()
F["high52"] = idx - idx.groupby(y.code).transform(lambda x: x.rolling(12, min_periods=10).max())
F["maxret"] = roll(r, 12, "max", 10)
F["skew"] = roll(r, 12, "skew", 10)
sea = sum(g.adjusted_monthly_return.shift(12 * k) for k in range(1, 6)) / 5
cnt = sum(g.adjusted_monthly_return.shift(12 * k).notna().astype(int) for k in range(1, 6))
F["season"] = (sum(g.adjusted_monthly_return.shift(12 * k).fillna(0) for k in range(1, 6)) / cnt.replace(0, np.nan)).where(cnt >= 3)
F["consist"] = roll((r > 0).astype(float).where(r.notna()), 12, "mean", 10)
F["price"] = np.log(y.end_price.where(y.end_price > 0))
# 시장 수익률(시총가중, 전월 시총) → 24개월 잔차 변동성
y["w"] = g.market_cap.shift(1)
mkt = (y.dropna(subset=["adjusted_monthly_return", "w"]).groupby("month").apply(lambda d: np.average(d.adjusted_monthly_return, weights=d.w), include_groups=False))
y["mkt"] = y.month.map(mkt)
def idv(d):
    out = np.full(len(d), np.nan); a = d.adjusted_monthly_return.values; m = d.mkt.values
    for i in range(len(d)):
        lo = max(0, i - 23); aa, mm = a[lo:i + 1], m[lo:i + 1]; ok = ~np.isnan(aa) & ~np.isnan(mm)
        if ok.sum() >= 18:
            b = np.polyfit(mm[ok], aa[ok], 1); out[i] = np.std(aa[ok] - np.polyval(b, mm[ok]), ddof=2)
    return pd.Series(out, index=d.index)
F["idvol"] = y.groupby("code", group_keys=False).apply(idv)
F["retail3"] = roll(y.retail_to_mcap, 3, "sum", 3)
F["foreign3"] = roll(y.foreign_to_mcap, 3, "sum", 3)
be = (y.BPS * y.shares_outstanding).where(y.BPS > 0)
F["bgrowth"] = np.log(be / be.groupby(y.code).shift(12))
cd = pd.read_csv("/home/user/data/RiskFree_Rate_CD91.csv"); cd.columns = ["m", "cd"]; cd["month"] = cd.m.str.replace("/", "-")
cd = cd.set_index("month").cd.astype(float); dcd = (cd - cd.shift(3))
F["dcd"] = F.month.map(dcd)
F.to_parquet("/home/user/fdi-team04-1-/연습장/out_yw/feat_cand.parquet"); print(F.shape); print(F.drop(columns=["month","code"]).describe().T[["count","mean","50%"]].round(3))
