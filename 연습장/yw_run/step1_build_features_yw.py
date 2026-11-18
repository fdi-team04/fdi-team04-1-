"""0단계: 예원 정형 데이터셋(final_structured_data_raw.csv) → 분석용 features.parquet
모든 정형 결과(4~7장)와 뉴스 분석(9·14장)의 수익률·통제 변수가 이 파일에서 나옴.
쓰는 법: python step1_build_features_yw.py <final_structured_data_raw.csv 경로>
변수 대응: beta=market_beta_24m, size=log(market_cap), bm=1/PBR(PBR>0), rev=adjusted_monthly_return(이번 달 수익률),
          mom=momentum_12_2, ivol(=변동성)=volatility_12m, issue=log(주식수/12개월 전 주식수),
          retail_nb=retail_to_mcap, foreign_nb=foreign_to_mcap, target=다음 달 수익률(예원 칸 그대로)
"""
import sys, numpy as np, pandas as pd
SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/yewon/x/final_structured_data_raw.csv"
OUT = "/home/user/fdi-team04-1-/연습장/out_yw/features.parquet"
y = pd.read_csv(SRC, dtype={"Stock_Code": str})
y["month"] = y.Month.str[:7]; y["code"] = y.Stock_Code.str.zfill(6)
y = y.sort_values(["code", "month"])
f = pd.DataFrame({"month": y.month, "code": y.code, "me": y.market_cap, "ret": y.adjusted_monthly_return, "target": y.target,
    "beta": y.market_beta_24m, "size": np.log(y.market_cap.where(y.market_cap > 0)), "bm": 1 / y.PBR.where(y.PBR > 0),
    "rev": y.adjusted_monthly_return, "mom": y.momentum_12_2, "ivol": y.volatility_12m,
    "issue": np.log(y.shares_outstanding / y.groupby("code").shares_outstanding.shift(12)),
    "retail_nb": y.retail_to_mcap, "foreign_nb": y.foreign_to_mcap})
f = f[f.ret.notna() | f.target.notna()]
if len(sys.argv) > 2 and sys.argv[2] == "--check":
    old = pd.read_parquet(OUT); print("기존과 같음:", old.reset_index(drop=True).equals(f.reset_index(drop=True)), old.shape, f.shape)
else:
    f.to_parquet(OUT); print(f.shape)
