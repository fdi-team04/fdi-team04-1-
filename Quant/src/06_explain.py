"""[06] 방향 확인과 해석: 정형 변수 방향, 목표가 상향(H3)·개인 매수(H4)·규모별 효과, SHAP

입력  data/processed/model_input.csv (config의 model_input) → 04의 prepare로 월별 순위 변환
출력  data/processed/explain_fm.csv     Fama-MacBeth 계수(%p)·t값 (analysis 열로 분석 구분)
      data/processed/explain_shap.csv   변수별 평균 |SHAP|(중요도)와 방향
      data/processed/explain_shap.png

기간: test 전 행(~2021-11)만 쓴다. FM의 통제 변수는 모두 정형 9개다.
  전체 종목: 정형 9개의 방향 (H1·H2 포함)
  H3: 수집 대상(상위 200) 안에서 tp_up 계수. 전체와 기간별(~2010, 2011–15, 2016–21)
  H4: 상위 200 안에서 tp_up × 개인 순매수 순위. tp_up = 1 종목이 3개 이상인 달만 (1~2개면 두 항이 거의 같아짐)
  규모: 상위 200을 매달 시총 3등분해 tp_up 계수 (교수님 11/23: 작은 기업에서 효과가 더 클 수 있음)
"""
import importlib.util
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt  # noqa: E402

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"
PERIODS = [("~2010", "2000-01", "2010-12"), ("2011-15", "2011-01", "2015-12"), ("2016-21", "2016-01", "2021-12")]


def _load_04():
    """04_model.py의 prepare·STAGES·make_model을 그대로 쓴다 (파일 이름이 숫자로 시작해 import 대신 경로로 읽음)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_model.py")
    spec = importlib.util.spec_from_file_location("model04", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _t(x: pd.Series) -> float:
    """월별 값의 평균 / 표준오차"""
    x = x.dropna()
    return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 1 else np.nan


def fama_macbeth(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """월별 횡단면 OLS → 계수 평균(%p), t = 평균 / 표준오차 (|t| ≥ 2)

    coef_pct = 다른 변수가 같을 때 그 변수가 0 → 1(순위 꼴찌 → 1등, 0/1 변수는 없음 → 있음)로 갈 때
               다음 달 수익률 차이(%p).
    그 달 값이 모두 같은 변수(예: 목표가 기사가 없는 달의 tp_up)는 그 달 회귀에서 빼고 계수를 결측으로 둔다.
    """
    rows = []
    for month, g in df.dropna(subset=["target"]).groupby("month"):
        use = [c for c in cols if g[c].nunique() > 1]
        if len(g) <= len(use) + 1:
            continue
        X = np.column_stack([np.ones(len(g)), g[use].to_numpy(float)])
        beta = np.linalg.lstsq(X, g["target"].to_numpy(float) * 100, rcond=None)[0]
        rows.append(dict(zip(use, beta[1:]), month=month))
    b = pd.DataFrame(rows).reindex(columns=["month"] + cols)
    return pd.DataFrame({"var": cols, "coef_pct": [b[c].mean() for c in cols],
                         "t": [_t(b[c]) for c in cols], "n_months": [b[c].notna().sum() for c in cols]})


def shap_importance(df: pd.DataFrame, cols: list[str], path: str | None = None) -> pd.DataFrame:
    """변수별 평균 |SHAP|(중요도, 큰 순서)와 방향(변수값과 SHAP 값의 상관: +면 값이 클수록 다음 달 순위↑)

    모델: test 전 행 전체로 한 번 학습한 xgb. 04와 같은 방식(make_model, y = rank_target, 입력 = feats).
    SHAP은 xgboost 내장 TreeSHAP(pred_contribs=True)으로 계산한다. 단위는 다음 달 수익률 순위(0~1).
    """
    import xgboost as xgb

    m04 = _load_04()
    X = m04.feats(df, cols)
    d = df.assign(y=m04.rank_target(df)).dropna(subset=["y"])
    model = m04.make_model("xgb").fit(d[X], d["y"])
    sv = model.get_booster().predict(xgb.DMatrix(d[X]), pred_contribs=True)[:, :-1]  # 마지막 열은 기준값
    imp = pd.DataFrame({"var": X, "mean_abs_shap": np.abs(sv).mean(axis=0),
                        "direction": [np.corrcoef(d[c], sv[:, k])[0, 1] for k, c in enumerate(X)]})
    imp = imp.sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    if path:
        r = imp[::-1]
        fig, ax = plt.subplots(figsize=(5.5, 0.3 * len(r) + 1))
        ax.barh(r["var"], r["mean_abs_shap"], color=np.where(r["direction"] > 0, "#4f81bd", "#c0504d"))
        ax.set_xlabel("mean |SHAP| (return-rank units), blue = higher value -> higher rank")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return imp


if __name__ == "__main__":
    m04 = _load_04()
    df = m04.prepare(m04.load_input(f"{cfg['data_dir']}/{cfg['model_input']}"))
    df = df[(df["month"] >= cfg["periods"]["train"][0]) & (df["month"] < m04.TEST_ROW)]  # test 행은 쓰지 않는다
    base = sum(m04.STAGES[:4], [])  # 정형 9개
    fm = [fama_macbeth(df, base).assign(analysis="전체 종목: 정형 9개")]
    if "tp_up" in df:
        top = df[df["news_missing"] == 0].copy()
        fm.append(fama_macbeth(top, base + ["tp_up"]).assign(analysis="H3 상위200 전체"))
        for name, a, b in PERIODS:
            sub = top[(top["month"] >= a) & (top["month"] <= b)]
            fm.append(fama_macbeth(sub, base + ["tp_up"]).assign(analysis=f"H3 상위200 {name}"))
        h4 = top[top.groupby("month")["tp_up"].transform("sum") >= 3]
        h4 = h4.assign(tp_x_retail=h4["tp_up"] * h4["retail_nb"])
        fm.append(fama_macbeth(h4, base + ["tp_up", "tp_x_retail"]).assign(analysis="H4 상위200: tp_up × 개인 순매수"))
        top["size3"] = top.groupby("month")["me"].transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=False))
        for k, g in top.groupby("size3"):
            fm.append(fama_macbeth(g, base + ["tp_up"]).assign(analysis=f"규모 {['작은', '중간', '큰'][int(k)]} 1/3 (상위200)"))
    else:
        print("[06] tp_up 열이 없어 H3·H4·규모별은 건너뜀 (목표가 변수를 아직 합치지 않은 파일)")
    fm = pd.concat(fm, ignore_index=True)[["analysis", "var", "coef_pct", "t", "n_months"]]
    fm.to_csv(f"{PROC}/explain_fm.csv", index=False, encoding="utf-8")
    key = (fm["analysis"] == "전체 종목: 정형 9개") | fm["var"].isin(["tp_up", "tp_x_retail"])
    print(f"[06] Fama-MacBeth (주요 행, 전체는 explain_fm.csv)\n{fm[key].round(3).to_string(index=False)}")

    imp = shap_importance(df, [c for c in sum(m04.STAGES, []) if c in df], f"{PROC}/explain_shap.png")
    imp.to_csv(f"{PROC}/explain_shap.csv", index=False, encoding="utf-8")
    print(f"[06] SHAP 중요도·방향\n{imp.round(4).to_string(index=False)}")
