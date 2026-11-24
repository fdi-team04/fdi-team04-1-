"""[04] 모델 입력 준비 → 확장 윈도우 학습·예측

입력  data/processed/model_input.csv     (config의 model_input: 예원 합친 파일 = v2 정형 + tp_up + news_missing)
출력  data/processed/predictions.parquet  (month, code, target, score, stage[, news_missing])
      data/processed/prediction.csv       (--final일 때만: test 행의 마지막 단계 xgb 점수)

준비(prepare): v2 열 → 모델 변수 9개, 월별 순위(0~1), 결측 0.5 + 변수명_missing. 06도 같은 함수를 쓴다.
STAGES: 단계마다 변수를 누적해 더한다. 기본은 xgb만 돌고, `--ridge`를 붙이면 선형 기준선(ridge)도 같이 돈다.
기간: 기본은 검증 행(2011-01 ~ 2021-11)만 예측한다. `--final`이면 test 행(2021-12 ~ 2025-12)까지 예측한다.
학습: 매년 1월 재학습. Y년 행은 학습 시작 ~ (Y−1)년 11월 행으로 학습한 모델 하나로 예측한다.
      (Y−1)년 11월 행의 target = 12월 수익률이라 Y년 1월 초에 이미 알 수 있다(미래 정보 아님).
"""
import os
import sys

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"
P = cfg["periods"]
TEST_ROW = str(pd.Period(P["test"][0], "M") - 1)  # "2021-12": 이 행부터 target(다음 달 수익률)이 test 기간이다

# 모델 변수 ← v2 열. 월별 순위만 쓰므로 log 같은 단조 변환은 하지 않는다(bm만 1/PBR로 방향을 맞춘다).
# vol은 수익률 전체의 표준편차(12개월)다. 시장모형 잔차가 아니므로 '고유변동성'이라 부르지 않는다.
FEATS = {"beta": "market_beta_24m", "size": "market_cap", "bm": "PBR",
         "rev": "adjusted_monthly_return", "mom": "momentum_12_2", "vol": "volatility_12m",
         "issue": "shares_outstanding_growth_12m", "retail_nb": "retail_to_mcap", "foreign_nb": "foreign_to_mcap"}
# v2가 같은 달 중앙값으로 채운 칸 → 빈칸으로 되돌려 다른 결측과 같은 규칙(0.5 + 표시 열)을 따르게 한다
IMPUTED = {"bm": "pbr_median_imputed", "issue": "shares_growth_median_imputed",
           "retail_nb": "investor_flow_median_imputed", "foreign_nb": "investor_flow_median_imputed"}

STAGES = [
    ["beta", "size", "bm"],         # 0 기준선
    ["rev", "mom"],                 # 1 통제
    ["vol", "issue"],               # 2 보강
    ["retail_nb", "foreign_nb"],    # 3 수급 (H1·H2)
    ["tp_up", "news_missing"],      # 4 뉴스 (H3): 목표가 상향 + 수집 대상 밖 표시 (0/1 그대로)
]


def load_input(path: str) -> pd.DataFrame:
    """예원 합친 파일 (csv 또는 parquet). 종목코드는 앞자리 0을 지키려고 문자열로 읽는다."""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"Stock_Code": str, "code": str})


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """v2 열 → month, code, target, me(시총, 원값), 모델 변수 9개(월별 순위), 변수명_missing, tp_up, news_missing

    규칙(계획서 2.3절): 순위 = 그달 값이 있는 종목 안의 백분위(rank(pct=True)), 결측 = 0.5 + 표시 열.
    그달 단면만 쓰므로 기간을 넘는 통계가 없다(test 누수 없음).
    뉴스 열이 빈 행(합치기에서 빠진 행)은 수집 대상 밖으로 본다: tp_up = 0, news_missing = 1.
    """
    month = raw["month"] if "month" in raw else raw["Month"]   # 합친 파일이 v2 열 이름을 그대로 써도 받는다
    code = raw["code"] if "code" in raw else raw["Stock_Code"]
    d = pd.DataFrame({"month": month.astype(str).str[:7], "code": code.astype(str).str.zfill(6),
                      "target": raw["target"], "me": raw["market_cap"]})
    x = pd.DataFrame({k: raw[v] for k, v in FEATS.items()})
    x["bm"] = 1 / x["bm"].where(x["bm"] > 0)
    for c, flag in IMPUTED.items():
        if flag in raw:
            x.loc[raw[flag] == 1, c] = np.nan
    miss = x.isna().astype(int).add_suffix("_missing")
    x = x.groupby(d["month"]).rank(pct=True).fillna(0.5)
    for c, fill in {"tp_up": 0, "news_missing": 1}.items():
        if c in raw:
            d[c] = raw[c].fillna(fill).astype(int)
    return pd.concat([d, x, miss], axis=1)


def make_model(name: str):
    """'xgb' → XGBRegressor(**cfg['xgb'], random_state=cfg['seed']), 'ridge' → Ridge"""
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(**cfg["xgb"], random_state=cfg["seed"])
    if name == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge()  # 변수가 이미 0~1 순위값이라 스케일링하지 않는다
    raise ValueError(f"모르는 모델: {name}")


def rank_target(df: pd.DataFrame) -> pd.Series:
    """학습용 y = 매달 종목 간 다음 달 수익률 순위 (0~1)

    IC가 순위만 보므로 순위를 학습한다 → 시장 전체 등락과 극단 수익률의 영향이 줄어든다.
    예측 파일의 target은 원래 수익률 그대로 둔다(05는 순위상관이라 결과가 같다).
    """
    return df.groupby("month")["target"].rank(pct=True)


def feats(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """모델 입력 = cols + 결측이 실제로 있는 변수의 변수명_missing (있는지는 test 전 행으로만 판단)"""
    tr = df[df["month"] < TEST_ROW]
    return cols + [f"{c}_missing" for c in cols if f"{c}_missing" in tr and tr[f"{c}_missing"].any()]


def expanding_predict(df: pd.DataFrame, cols: list[str], name: str, months: list[str]) -> pd.DataFrame:
    """months 행을 해마다 한 모델로 예측한다. 출력: month, code, target, score[, news_missing]

    Y년 모델 = 학습 시작 ~ (Y−1)년 11월 행으로 학습 (y = rank_target, 입력 = feats). target이 없는 행은 학습에서 뺀다.
    """
    X = feats(df, cols)
    train = df.assign(y=rank_target(df)).dropna(subset=["y"])
    keep = [c for c in ["month", "code", "target", "news_missing"] if c in df]
    te = df[df["month"].isin(months)]
    out = []
    for year, g in te.groupby(te["month"].str[:4]):
        tr = train[train["month"] <= f"{int(year) - 1}-11"]
        out.append(g[keep].assign(score=make_model(name).fit(tr[X], tr["y"]).predict(g[X])))
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    final = "--final" in sys.argv
    names = ["xgb", "ridge"] if "--ridge" in sys.argv else ["xgb"]
    df = prepare(load_input(f"{cfg['data_dir']}/{cfg['model_input']}"))
    df = df[df["month"] >= P["train"][0]]  # warmup 제외
    months = [m for m in sorted(df["month"].unique())
              if P["valid"][0] <= m and (m <= P["test"][1] if final else m < TEST_ROW)]
    cols, parts = [], []
    for i, add in enumerate(STAGES):
        if any(c not in df for c in add):
            print(f"[04] {add} 열이 없어 {i}단계부터 건너뜀 (목표가 변수를 아직 합치지 않은 파일)")
            break
        cols, last = cols + add, f"xgb/{i}"
        for name in names:
            parts.append(expanding_predict(df, cols, name, months).assign(stage=f"{name}/{i}"))
            print(f"[04] {name}/{i} 완료: 변수 {cols}")
    pred = pd.concat(parts, ignore_index=True)
    os.makedirs(PROC, exist_ok=True)
    pred.to_parquet(f"{PROC}/predictions.parquet", index=False)
    if final:  # 제출용: test 행만, 마지막 단계 xgb 점수
        sub = pred[(pred["stage"] == last) & (pred["month"] >= TEST_ROW)]
        sub[["month", "code", "score"]].to_csv(f"{PROC}/prediction.csv", index=False)
        print(f"[04] prediction.csv 저장: {last}, {sub['month'].min()} ~ {sub['month'].max()}, {len(sub):,}행")
