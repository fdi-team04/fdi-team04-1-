"""[EDA] 타겟 생성 + 변수–타겟 관계 확인   담당: 위영섭

데이터 열어보기 분담 1번(11/5):
  - 타겟(다음 달 조정수익률) 생성
  - 규모 · 단기 반전 · 모멘텀과 타겟의 관계 그래프
  - 월별 종목 수, 빈 값 비율, 수익률 분포 확인

입력  은예원 long format 파일 (warmup_train_long.csv 또는 panel.parquet)
      필요한 열: Month, Stock_Code, adjusted_monthly_return, market_cap
출력  <out>/ 아래 그림(png)과 표(csv), summary.md

실행  python Quant/eda/target_and_correlation.py <long 파일 경로> [--out 출력 폴더]

규칙
  - test(2022-01 이후) 행은 읽자마자 버린다. 이 스크립트는 2021-12까지만 본다.
  - 종목코드는 6자리 문자열로 맞춘다(앞자리 0 유지).
  - 빈 값은 채우지 않는다. 다음 달 행이 없으면 타겟은 결측이다(0으로 보지 않음).
  - 이 결과는 '방향 미리 보기'다. 변수 생존 판정은 단계별 IC 표(05)로 한다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LAST_TRAIN_MONTH = pd.Period("2021-12", freq="M")
MOM_WINDOW = 11          # t-11 ~ t-1 (t월은 단기 반전이 맡음)
MIN_STOCKS_FOR_IC = 30   # 이보다 종목이 적은 달은 IC를 계산하지 않음
VARS = {
    "size": "규모 = ln(t월 말 시가총액)  · 예상 방향 −",
    "rev": "단기 반전 = t월 수익률  · 예상 방향 −",
    "mom": "모멘텀 = t−11~t−1월 누적 수익률  · 방향은 데이터로 확인",
}


# ------------------------------------------------------------------
# 1. 불러오기
# ------------------------------------------------------------------
def load_long(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"Stock_Code": str})

    df = df.rename(columns={c: "market_cap" for c in df.columns
                            if c.lower() in ("raw_month_end_market_cap", "month_end_market_cap")})
    cols = {c.lower(): c for c in df.columns}
    need = ["month", "stock_code", "adjusted_monthly_return", "market_cap"]
    missing = [n for n in need if n not in cols]
    if missing:
        raise ValueError(f"필요한 열이 없습니다: {missing}. 실제 열: {list(df.columns)}")
    df = df.rename(columns={cols[n]: n for n in need})

    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")
    n_before = len(df)
    df = df[df["month"] <= LAST_TRAIN_MONTH].copy()   # test 차단
    dropped = n_before - len(df)

    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.zfill(6)
    for c in ["adjusted_monthly_return", "market_cap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["stock_code", "month"]).reset_index(drop=True)
    df.attrs["dropped_test_rows"] = dropped
    return df


def detect_return_unit(r: pd.Series) -> float:
    """수익률이 % 단위(예: 3.5)인지 소수(0.035)인지 판단해 소수로 바꿀 배율을 돌려준다."""
    med_abs = r.abs().median()
    return 0.01 if med_abs > 0.5 else 1.0


# ------------------------------------------------------------------
# 2. 타겟과 변수
# ------------------------------------------------------------------
def add_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    scale = detect_return_unit(df["adjusted_monthly_return"].dropna())
    df["ret"] = df["adjusted_monthly_return"] * scale
    df.attrs["return_scale"] = scale

    g = df.groupby("stock_code", sort=False)
    # 바로 다음 달 행일 때만 타겟으로 인정 (중간에 빠진 달이 있으면 결측)
    next_month = g["month"].shift(-1)
    consecutive = next_month == (df["month"] + 1)
    df["target"] = g["ret"].shift(-1).where(consecutive)

    df["size"] = np.log(df["market_cap"].where(df["market_cap"] > 0))
    df["rev"] = df["ret"]

    # 모멘텀: t-11 ~ t-1, 11개월이 모두 있고 달이 이어질 때만
    log1p = np.log1p(df["ret"].where(df["ret"] > -1))
    lagged = g["month"].shift(1)
    roll = log1p.groupby(df["stock_code"]).transform(
        lambda s: s.shift(1).rolling(MOM_WINDOW, min_periods=MOM_WINDOW).sum()
    )
    start = g["month"].shift(MOM_WINDOW)
    contiguous = (start == (df["month"] - MOM_WINDOW)) & (lagged == df["month"] - 1)
    df["mom"] = np.expm1(roll).where(contiguous)
    return df


# ------------------------------------------------------------------
# 3. 데이터 상태 점검
# ------------------------------------------------------------------
def data_checks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly = df.groupby("month").agg(
        rows=("stock_code", "size"),
        with_return=("ret", "count"),
        with_target=("target", "count"),
        with_mcap=("market_cap", "count"),
    )
    monthly["return_missing_rate"] = 1 - monthly["with_return"] / monthly["rows"]

    cols = ["ret", "market_cap", "target", "size", "rev", "mom"]
    missing = pd.DataFrame({
        "missing_count": df[cols].isna().sum(),
        "missing_rate": df[cols].isna().mean(),
    })

    q = [0, .001, .01, .05, .25, .5, .75, .95, .99, .999, 1]
    dist = df["ret"].quantile(q).to_frame("monthly_return")
    dist.index = [f"p{x*100:g}" for x in q]

    flags = pd.Series({
        "ret <= -100% (불가능)": int((df["ret"] <= -1).sum()),
        "ret > +100%": int((df["ret"] > 1).sum()),
        "market_cap <= 0": int((df["market_cap"] <= 0).sum()),
        "month+code 중복": int(df.duplicated(["month", "stock_code"]).sum()),
    }).to_frame("count")
    return {"monthly_counts": monthly, "missing": missing, "return_dist": dist, "flags": flags}


# ------------------------------------------------------------------
# 4. 변수–타겟 관계: 월별 순위상관(IC)과 5분위 수익률
# ------------------------------------------------------------------
def monthly_ic(df: pd.DataFrame, var: str) -> pd.Series:
    def one(m: pd.DataFrame) -> float:
        m = m[[var, "target"]].dropna()
        if len(m) < MIN_STOCKS_FOR_IC:
            return np.nan
        return m[var].rank().corr(m["target"].rank())
    return df.groupby("month")[[var, "target"]].apply(one).rename(var)


def ic_summary(ics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in ics.columns:
        s = ics[v].dropna()
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
        rows.append({
            "variable": v, "mean_ic": s.mean(), "t_stat": t,
            "pos_ratio": (s > 0).mean(), "n_months": len(s),
            "first": str(s.index.min()), "last": str(s.index.max()),
        })
    return pd.DataFrame(rows).set_index("variable")


def quintile_returns(df: pd.DataFrame, var: str) -> pd.Series:
    d = df[["month", var, "target"]].dropna().copy()
    d["q"] = d.groupby("month")[var].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= MIN_STOCKS_FOR_IC else np.nan
    )
    by_month = d.dropna(subset=["q"]).groupby(["month", "q"])["target"].mean().unstack()
    return by_month.mean()  # 5분위별 '다음 달 평균 수익률'의 월평균 (단순 평균)


def ic_by_year(ics: pd.DataFrame) -> pd.DataFrame:
    y = ics.copy()
    y.index = y.index.year
    return y.groupby(level=0).mean()


# ------------------------------------------------------------------
# 5. 그림
# ------------------------------------------------------------------
def make_figures(df, checks, ics, quint, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    labels = {"size": "Size  ln(mcap)", "rev": "Short-term reversal  r(t)", "mom": "Momentum  r(t-11..t-1)"}

    # (1) 월별 종목 수
    fig, ax = plt.subplots(figsize=(9, 3.2))
    mc = checks["monthly_counts"]
    x = mc.index.to_timestamp()
    ax.plot(x, mc["with_return"], label="stocks with return", lw=1.5)
    ax.plot(x, mc["with_target"], label="stocks with next-month target", lw=1, ls="--")
    ax.set_title("Monthly number of stocks (2000-01 ~ 2021-12)")
    ax.legend(frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); f = out / "fig1_monthly_counts.png"; fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)

    # (2) 수익률 분포
    fig, ax = plt.subplots(figsize=(6, 3.2))
    r = df["ret"].dropna().clip(-0.6, 1.0)
    ax.hist(r, bins=120, color="#1D6A86")
    ax.set_yscale("log"); ax.set_title("Monthly return distribution (clipped to -60%..+100%)")
    ax.set_xlabel("monthly return"); ax.grid(alpha=.3)
    fig.tight_layout(); f = out / "fig2_return_distribution.png"; fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)

    # (3) 5분위별 다음 달 평균 수익률
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, v in zip(axes, ["size", "rev", "mom"]):
        qv = quint[v] * 100
        ax.bar(qv.index.astype(int).astype(str), qv.values, color="#1D6A86")
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(labels[v], fontsize=10); ax.set_xlabel("quintile (1 = low, 5 = high)")
        ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("avg next-month return (%)")
    fig.suptitle("Next-month return by quintile (equal-weighted, monthly average)", fontsize=11)
    fig.tight_layout(); f = out / "fig3_quintile_returns.png"; fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)

    # (4) 누적 IC
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for v in ["size", "rev", "mom"]:
        s = ics[v].fillna(0).cumsum()
        ax.plot(s.index.to_timestamp(), s.values, label=labels[v], lw=1.4)
    ax.axhline(0, color="k", lw=.6)
    ax.set_title("Cumulative monthly Spearman IC (falling line = negative relation)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); f = out / "fig4_cumulative_ic.png"; fig.savefig(f, dpi=150); plt.close(fig); files.append(f.name)
    return files


# ------------------------------------------------------------------
# 6. 요약
# ------------------------------------------------------------------
def write_summary(df, checks, summ, quint, yearly, figs, out: Path) -> None:
    mc = checks["monthly_counts"]
    lines = [
        "# 데이터 열어보기 1번: 타겟 + 규모·단기 반전·모멘텀",
        "",
        f"- 기간: {df['month'].min()} ~ {df['month'].max()} (test 행 {df.attrs['dropped_test_rows']}개는 읽자마자 제외)",
        f"- 수익률 단위: {'% 단위 → 소수로 변환' if df.attrs['return_scale'] != 1 else '소수'}",
        f"- 종목 수(수익률 있는 종목): 월평균 {mc['with_return'].mean():.0f}, 최소 {mc['with_return'].min()} ({mc['with_return'].idxmin()}), 최대 {mc['with_return'].max()} ({mc['with_return'].idxmax()})",
        f"- 타겟이 있는 행: {df['target'].notna().sum():,} / 수익률이 있는 행 {df['ret'].notna().sum():,}",
        "",
        "## 이상값 점검",
        checks["flags"].to_markdown(),
        "",
        "## 빈 값 비율",
        checks["missing"].to_markdown(floatfmt=".3f"),
        "",
        "## 수익률 분포",
        checks["return_dist"].to_markdown(floatfmt=".4f"),
        "",
        "## 변수와 다음 달 수익률의 월별 순위상관(IC)",
        "IC < 0 이면 값이 클수록 다음 달 순위가 낮다는 뜻. |t| ≥ 2 면 방향이 꾸준하다고 본다.",
        "",
        summ.to_markdown(floatfmt=".4f"),
        "",
        "## 5분위별 다음 달 평균 수익률 (단순 평균, 소수)",
        quint.T.to_markdown(floatfmt=".4f"),
        "",
        "## 연도별 평균 IC",
        yearly.to_markdown(floatfmt=".3f"),
        "",
        "## 그림",
        *[f"- {f}" for f in figs],
        "",
        "## 해석할 때 주의",
        "- 이 표는 변수 하나씩 본 '방향 미리 보기'다. 다른 변수를 함께 넣은 효과는 단계별 IC 표와 Fama-MacBeth로 본다.",
        "- 5분위 평균은 단순 평균이다(시가총액 가중 여부는 팀 미정 사항).",
        f"- 종목이 {MIN_STOCKS_FOR_IC}개 미만인 달은 IC를 계산하지 않았다.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eda_out/target_corr"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = add_target_and_features(load_long(a.path))
    checks = data_checks(df)
    ics = pd.concat([monthly_ic(df, v) for v in VARS], axis=1)
    summ = ic_summary(ics)
    quint = pd.DataFrame({v: quintile_returns(df, v) for v in VARS})
    yearly = ic_by_year(ics)

    for name, t in checks.items():
        t.to_csv(a.out / f"check_{name}.csv", encoding="utf-8-sig")
    ics.to_csv(a.out / "ic_monthly.csv", encoding="utf-8-sig")
    summ.to_csv(a.out / "ic_summary.csv", encoding="utf-8-sig")
    quint.to_csv(a.out / "quintile_returns.csv", encoding="utf-8-sig")
    yearly.to_csv(a.out / "ic_by_year.csv", encoding="utf-8-sig")
    df[["month", "stock_code", "ret", "target", "size", "rev", "mom"]].assign(month=lambda d: d["month"].astype(str)) \
        .to_csv(a.out / "target_features.csv", index=False, encoding="utf-8-sig")

    figs = make_figures(df, checks, ics, quint, a.out)
    write_summary(df, checks, summ, quint, yearly, figs, a.out)
    print(summ.round(4).to_string())
    print(f"\n결과: {a.out.resolve()}")


if __name__ == "__main__":
    main()
