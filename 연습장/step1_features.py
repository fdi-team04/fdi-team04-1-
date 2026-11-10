"""1단계: 변수 만들기 (t월 말까지 알 수 있는 정보만) + target = t+1월 수익률
입력: joined_investor.csv, monthly_pbr_2000_2025.csv, RiskFree_Rate_CD91.csv
출력: out/features.parquet
"""
import numpy as np, pandas as pd, sys
D = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data"
OUT = "/home/user/fdi-team04-1-/연습장/out"
import os; os.makedirs(OUT, exist_ok=True)

j = pd.read_csv(f"{D}/joined_investor.csv", dtype={"Stock_Code": str})
j["month"] = j["Month"].str[:7]
j["code"] = j["Stock_Code"].str.zfill(6)
j = j[j.month <= "2021-12"]                     # test(2022~) 읽자마자 제외
df = j.rename(columns={"adjusted_monthly_return": "ret", "market_cap": "me",
                       "shares_outstanding": "shares",
                       "retail_to_mcap": "retail_nb", "foreign_to_mcap": "foreign_nb"})
df = df[["month", "code", "ret", "me", "shares", "retail_nb", "foreign_nb"]]
df = df.sort_values(["code", "month"]).reset_index(drop=True)

months = sorted(df.month.unique()); midx = {m: i for i, m in enumerate(months)}
df["mi"] = df.month.map(midx)
g = df.groupby("code")

# ---- target: 다음 달 수익률 (바로 다음 달 행일 때만)
nxt_ret = g.ret.shift(-1); nxt_mi = g.mi.shift(-1)
df["target"] = np.where(nxt_mi == df.mi + 1, nxt_ret, np.nan)

# ---- 무위험수익률(CD91 연%, 월로 환산), 시장수익률(직전 달 시총 가중)
rf = pd.read_csv(f"{D}/RiskFree_Rate_CD91.csv", encoding="utf-8-sig")
rf.columns = ["m", "rate"]; rf["month"] = rf.m.str.replace("/", "-"); rf["rf"] = rf.rate.astype(float) / 100 / 12
df = df.merge(rf[["month", "rf"]], on="month", how="left")
df["me_l"] = df.groupby("code").me.shift(1)
w = df.dropna(subset=["ret", "me_l"])
mkt = w.groupby("month").apply(lambda x: np.average(x.ret, weights=x.me_l), include_groups=False).rename("mkt")
df = df.merge(mkt, on="month", how="left")
df["ex"] = df.ret - df.rf; df["mex"] = df.mkt - df.rf

# ---- 기준선: beta(60개월, 최소 24), size, bm(=1/PBR, PBR<=0은 결측)
def roll_beta(s_ex, s_m, win=60, minp=24):
    cov = s_ex.rolling(win, minp).cov(s_m); var = s_m.rolling(win, minp).var()
    return cov / var
df = df.sort_values(["code", "month"]).reset_index(drop=True)
out = []
for c, x in df.groupby("code", sort=False):
    b = roll_beta(x.ex, x.mex)
    # 고유변동성: 최근 24개월(최소 12) 시장모형 잔차 표준편차
    b24 = roll_beta(x.ex, x.mex, 24, 12)
    a24 = x.ex.rolling(24, 12).mean() - b24 * x.mex.rolling(24, 12).mean()
    # 창 안의 잔차: ex - a - b*mex 를 창 끝 계수로 계산 → 분산 공식으로 대체
    var_e = x.ex.rolling(24, 12).var() - b24**2 * x.mex.rolling(24, 12).var()
    ivol = np.sqrt(var_e.clip(lower=0))
    # 모멘텀: t-11 ~ t-1 누적수익률 (t월 제외), 11개월 중 8개월 이상
    lr = np.log1p(x.ret)
    mom = np.expm1(lr.shift(1).rolling(11, 8).sum())
    issue = np.log(x.shares / x.shares.shift(12))
    out.append(pd.DataFrame({"beta": b, "ivol": ivol, "mom": mom, "issue": issue}, index=x.index))
df = df.join(pd.concat(out))
df["size"] = np.log(df.me)
df["rev"] = df.ret

p = pd.read_csv(f"{D}/monthly_pbr_2000_2025.csv", dtype={"ticker": str})
p["code"] = p.ticker.str.zfill(6)
p = p[p.month <= "2021-12"].drop_duplicates(["month", "code"])
p["bm"] = np.where(p.PBR > 0, 1 / p.PBR, np.nan)
df = df.merge(p[["month", "code", "bm"]], on=["month", "code"], how="left")

# 분석 대상: t월 수익률이 있는(상장된) 종목
df = df[df.ret.notna()].copy()
cols = ["month", "code", "me", "ret", "target", "beta", "size", "bm", "rev", "mom", "ivol", "issue",
        "retail_nb", "foreign_nb"]
df = df[cols].sort_values(["month", "code"]).reset_index(drop=True)
df.to_parquet(f"{OUT}/features.parquet", index=False)

# 변수별 채워진 비율(연도별)
cov = df.groupby(df.month.str[:4])[cols[4:]].apply(lambda x: x.notna().mean()).round(3)
cov.to_csv(f"{OUT}/coverage_by_year.csv")
print(df.shape); print(cov.to_string())
