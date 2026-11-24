"""[05] 평가: 월별 Spearman IC와 단계 판정   담당: 정아영

입력  data/processed/predictions.parquet  (04: month, code, target, score, stage[, news_missing])
출력  data/processed/ic_monthly.csv   (universe, period, stage, month, ic, n_stocks)
      data/processed/ic_summary.csv   (universe, period, stage, mean_ic, pos_ratio, min_ic, n_months)
      data/processed/ic_by_year.csv   (universe, period, stage, year, mean_ic)
      data/processed/ic_judge.csv     (universe, stage, vs, mean_ic, diff, t, years_up, pass) 검증 구간만
      data/processed/ic_valid.png     xgb 단계별 누적 IC·연평균 IC 그림 (--final이면 ic_test.png도)

평가 규칙(공지): 매월 "그 달 실제 수익률(target)이 있는 모든 종목"에서 예측값(score)과
실제 수익률의 Spearman 순위상관을 계산한다. 단계(stage)별로 따로 계산한다.
universe: all = 전체 종목, top200 = 뉴스 수집 대상(news_missing = 0). 뉴스 변수 효과는 top200에서 따로 본다.
period: valid = 검증 행(~2021-11), test = test 행(2021-12~, 04 --final일 때만 있음). 두 구간은 따로 요약한다.
"""
import os
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt  # noqa: E402

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"
TEST_ROW = str(pd.Period(cfg["periods"]["test"][0], "M") - 1)  # 04와 같은 경계: 이 행부터 test
MIN_STOCKS = 10    # 이보다 종목이 적은 달은 ic를 결측으로 둔다
MIN_YEARS_UP = 6   # 검증 11년 중 6년 이상 연평균 IC가 올라야 통과 (보고서 3.1 조건 3)
# 그림: 단계마다 고정 색(순서대로, 돌려 쓰지 않음)과 이름. 이름은 04 STAGES 순서이고 글꼴 문제로 영문
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STAGE_NAMES = ["base", "+rev, mom", "+vol, issue", "+retail, foreign", "+tp_up"]


def load_predictions(path: str) -> pd.DataFrame:
    """예측 파일 → target·score가 있는 행 + period, universe(all, 뉴스 열이 있으면 top200도)"""
    pred = pd.read_parquet(path).dropna(subset=["target", "score"])
    pred["month"] = pred["month"].astype(str).str[:7]
    pred["period"] = np.where(pred["month"] < TEST_ROW, "valid", "test")
    parts = [pred.assign(universe="all")]
    if "news_missing" in pred:
        parts.append(pred[pred["news_missing"] == 0].assign(universe="top200"))
    return pd.concat(parts, ignore_index=True)


def monthly_ic(pred: pd.DataFrame) -> pd.DataFrame:
    """(universe, period, stage, month)마다 score와 target의 Spearman 순위상관 (동순위는 평균 순위)

    시점: score는 t월 말 정보로 만든 값이고 target은 t+1월 수익률이다(미래 정보 아님).
    """
    rows = []
    for (uni, period, stage, month), g in pred.groupby(["universe", "period", "stage", "month"]):
        ic = g["score"].rank().corr(g["target"].rank()) if len(g) >= MIN_STOCKS else np.nan
        rows.append({"universe": uni, "period": period, "stage": stage, "month": month, "ic": ic, "n_stocks": len(g)})
    return pd.DataFrame(rows)

def ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """평균 IC, IC가 양(+)인 달의 비율, 가장 낮은 IC(급반등 국면 점검), 달 수"""
    g = ic.dropna(subset=["ic"]).groupby(["universe", "period", "stage"])["ic"]
    return g.agg(mean_ic="mean", pos_ratio=lambda v: (v > 0).mean(), min_ic="min", n_months="size").reset_index()

def ic_by_year(ic: pd.DataFrame) -> pd.DataFrame:
    """연도별 평균 IC (판정의 '오른 해' 계산과 같은 표)"""
    d = ic.dropna(subset=["ic"]).assign(year=lambda x: x["month"].str[:4])
    return d.groupby(["universe", "period", "stage", "year"])["ic"].mean().reset_index(name="mean_ic")


def judge(ic: pd.DataFrame) -> pd.DataFrame:
    """검증 구간에서 같은 모델의 직전 단계와 비교 (보고서 3.1). test는 판정에 쓰지 않는다.

    diff = 월별 IC 차이의 평균, t = 그 평균 / 표준오차, years_up = 연평균 IC가 직전 단계보다 높은 해 수.
    pass = diff > 0 이고 years_up ≥ MIN_YEARS_UP.
    """
    v = ic[ic["period"] == "valid"].dropna(subset=["ic"])
    v = v.assign(model=v["stage"].str.split("/").str[0])
    rows = []
    for (uni, _), g in v.groupby(["universe", "model"]):
        w = g.pivot(index="month", columns="stage", values="ic")
        w = w[sorted(w.columns, key=lambda s: int(s.split("/")[1]))]
        yr = w.groupby(w.index.str[:4]).mean()
        for prev, cur in zip(w.columns[:-1], w.columns[1:]):
            d = (w[cur] - w[prev]).dropna()
            up = int((yr[cur] > yr[prev]).sum())
            rows.append({"universe": uni, "stage": cur, "vs": prev, "mean_ic": w[cur].mean(), "diff": d.mean(),
                         "t": d.mean() / (d.std() / np.sqrt(len(d))), "years_up": f"{up}/{len(yr)}",
                         "pass": bool(d.mean() > 0 and up >= MIN_YEARS_UP)})
    return pd.DataFrame(rows)


def plot_ic(ic: pd.DataFrame, period: str, path: str) -> None:
    """xgb 단계별 IC 그림. 왼쪽 = 월별 IC 누적합(꾸준히 오르는지), 오른쪽 = 연평균 IC(판정의 '오른 해'). 행 = universe"""
    d = ic[(ic["period"] == period) & ic["stage"].str.startswith("xgb/")].dropna(subset=["ic"])
    unis = list(d["universe"].unique())
    fig, axes = plt.subplots(len(unis), 2, figsize=(10, 3.3 * len(unis)), squeeze=False)
    for r, uni in enumerate(unis):
        w = d[d["universe"] == uni].pivot(index="month", columns="stage", values="ic")
        yr = w.groupby(w.index.str[:4]).mean()
        for s in sorted(w.columns, key=lambda x: int(x.split("/")[1])):
            i = int(s.split("/")[1])
            style = dict(color=COLORS[i], lw=2, label=f"{i} {STAGE_NAMES[i]}" if i < len(STAGE_NAMES) else s)
            axes[r, 0].plot(pd.to_datetime(w.index), w[s].cumsum(), **style)
            axes[r, 1].plot(yr.index.astype(int), yr[s], **style)
        axes[r, 0].set_title(f"{uni}: cumulative monthly IC", loc="left", fontsize=10, color="#0b0b0b")
        axes[r, 1].set_title(f"{uni}: yearly mean IC", loc="left", fontsize=10, color="#0b0b0b")
        for ax in axes[r]:
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.grid(color="#e6e5e1", lw=0.8)
            ax.tick_params(colors="#52514e", labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=len(STAGE_NAMES), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ic = monthly_ic(load_predictions(f"{PROC}/predictions.parquet"))
    out = {"ic_monthly": ic, "ic_summary": ic_summary(ic), "ic_by_year": ic_by_year(ic), "ic_judge": judge(ic)}
    os.makedirs(PROC, exist_ok=True)
    for name, table in out.items():
        table.to_csv(f"{PROC}/{name}.csv", index=False, encoding="utf-8")
    for period in ic["period"].unique():
        plot_ic(ic, period, f"{PROC}/ic_{period}.png")
    print(out["ic_summary"].round(4).to_string(index=False))
    print(out["ic_judge"].round(4).to_string(index=False))
