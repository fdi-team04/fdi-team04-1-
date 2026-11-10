"""2단계: XGBoost 누적 학습(확장 윈도우, 매년 1월 재학습) 단계별 IC + 판정

학습 방식
  검증 연도 Y = 2011 ... 2021
  학습 데이터: month 2001-01 ~ (Y-1)-11  (그 행의 target = 다음 달 수익률이 Y-1년 12월까지라 Y년 1월 초에 이미 알 수 있음)
  예측: Y년 1~12월 각 달의 종목 (t월 말 변수 → t+1월 수익률)
입력 변환: 매달 단면 순위(0~1), 빈칸은 0.5 + 결측 표시 열
타깃: 매달 단면 순위(0~1) — IC가 순위 기준이라 극단 수익률 영향 줄임
하이퍼파라미터: config.yaml 값 고정 (조정하지 않음)
판정: 직전 단계보다 평균 IC↑ 그리고 11개 검증 연도 중 6년 이상 연평균 IC↑ → OK
"""
import numpy as np, pandas as pd, matplotlib, sys
matplotlib.use("Agg"); import matplotlib.pyplot as plt, koreanize_matplotlib  # noqa
plt.rcParams["axes.unicode_minus"] = False
from xgboost import XGBRegressor
OUT = "/home/user/fdi-team04-1-/연습장/out"
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42,
           tree_method="hist", n_jobs=-1)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]

df = pd.read_parquet(f"{OUT}/features.parquet")
df = df[df.month >= "2001-01"].copy()                       # 2000년은 warmup: 학습·평가 제외
for c in ALL:
    df[c + "_m"] = df[c].isna().astype(float)
    df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
df["y"] = df.groupby("month").target.rank(pct=True)

def feats(cols):
    f = [c + "_r" for c in cols]
    f += [c + "_m" for c in cols if c in ("beta", "bm", "mom", "ivol", "issue")]   # 결측이 실제로 있는 변수만
    return f

def expanding(cols, shuffle=False):
    preds = []
    F = feats(cols)
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]
        te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        y = tr.y.values
        if shuffle:
            y = tr.groupby("month").y.transform(lambda s: s.sample(frac=1, random_state=0).values).values
        m = XGBRegressor(**XGB).fit(tr[F], y)
        preds.append(te[["month", "code", "target", "me"]].assign(score=m.predict(te[F])))
    return pd.concat(preds)

def monthly_ic(p):
    return p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)

def summarize(name, ic, prev_ic=None):
    yr = ic.groupby(ic.index.str[:4]).mean()
    r = {"단계": name, "평균IC": ic.mean(), "t": ic.mean() / (ic.std() / np.sqrt(len(ic))),
         "IC>0 비율": (ic > 0).mean(), "최저IC": ic.min(), "달수": len(ic)}
    if prev_ic is not None:
        d = ic - prev_ic; pyr = prev_ic.groupby(prev_ic.index.str[:4]).mean()
        wins = int((yr > pyr).sum())
        r.update({"증가분": d.mean(), "증가분t": d.mean() / (d.std() / np.sqrt(len(d))), "이긴 해": f"{wins}/11",
                  "판정": "OK" if (d.mean() > 0 and wins >= 6) else "NO"})
    return r, yr

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    res, yrs, ics = [], {}, {}
    # (A) 묶음 단계: 기준선 → +통제 → +보강 → +수급(H1·H2)
    STAGES = [("0 기준선", ["beta", "size", "bm"]),
              ("1 +통제(반전·모멘텀)", ["rev", "mom"]),
              ("2 +보강(고유변동성·발행)", ["ivol", "issue"]),
              ("3 +수급(개인·외국인)", ["retail_nb", "foreign_nb"])]
    cols, prev = [], None
    for name, add in STAGES:
        cols = cols + add
        p = expanding(cols); ic = monthly_ic(p); ics[name] = ic
        p.to_parquet(f"{OUT}/pred_{name.split()[0]}.parquet")
        r, yr = summarize(name, ic, prev); res.append(r); yrs[name] = yr; prev = ic
        print(r, flush=True)
    base3 = cols[:-2]
    # (B) 수급 하나씩: 2단계 + 개인만 / + 외국인만
    for name, add in [("3a +개인만", ["retail_nb"]), ("3b +외국인만", ["foreign_nb"])]:
        ic = monthly_ic(expanding(base3 + add)); ics[name] = ic
        r, yr = summarize(name, ic, ics["2 +보강(고유변동성·발행)"]); res.append(r); yrs[name] = yr; print(r, flush=True)
    # (C) 기준선 안에서 하나씩 쌓기 + 통제 하나씩
    seq = [("s1 베타", ["beta"]), ("s2 +규모", ["size"]), ("s3 +가치", ["bm"]), ("s4 +모멘텀", ["mom"]),
           ("s5 +단기반전", ["rev"]), ("s6 +고유변동성", ["ivol"]), ("s7 +주식발행", ["issue"])]
    cols, prev = [], None
    for name, add in seq:
        cols = cols + add
        ic = monthly_ic(expanding(cols)) if name not in ("s5 +단기반전", "s7 +주식발행") else None
        if name == "s5 +단기반전": ic = ics["1 +통제(반전·모멘텀)"]
        if name == "s7 +주식발행": ic = ics["2 +보강(고유변동성·발행)"]
        ics[name] = ic
        r, yr = summarize(name, ic, prev); res.append(r); yrs[name] = yr; prev = ic; print(r, flush=True)
    # (D) 누수 점검: 정답을 달마다 섞어서 학습 → IC ≈ 0 이어야 함
    ic = monthly_ic(expanding(base3 + ["retail_nb", "foreign_nb"], shuffle=True)); ics["누수점검(정답 섞기)"] = ic
    r, _ = summarize("누수점검(정답 섞기)", ic); res.append(r); print(r, flush=True)

    R = pd.DataFrame(res); R.to_csv(f"{OUT}/stage_ic.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yrs).to_csv(f"{OUT}/stage_ic_by_year.csv", encoding="utf-8-sig")
    pd.DataFrame(ics).to_csv(f"{OUT}/stage_ic_monthly.csv", encoding="utf-8-sig")
    print(R.round(4).to_string(index=False))
