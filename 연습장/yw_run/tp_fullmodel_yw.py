"""전체 코스피 모델에 목표가 상향(tp_up) 추가 → IC가 오르나 (정형 4단계와 같은 방식)
기준 모델: 정형 9개 변수 (step3_xgb_stages.py 4단계와 동일: 월별 순위 + 결측 표시, XGBoost, 확장 학습, 2011–2021 매년 1월 재학습)
비교 모델: 기준 + tp_up(목표가 상향 기사 있음 1/없음 0) + news_missing(전년 말 시총 상위 200 밖 = 1)
뉴스 변수: out/목표가상향_변수_합치기용.csv 를 month·code 왼쪽 합치기, 없는 행 = news_missing 1, tp_up 0 (월 이동 없음)
통과 기준(미리 정함): 평균 IC 상승 + 11년 중 6년 이상 연평균 IC 상승
참고: 실적 호재 v1(n_earn_up_v1>0)도 같은 방식으로 (판정은 보류 테마라 참고만)
"""
import numpy as np, pandas as pd
from xgboost import XGBRegressor
B = "/home/user/fdi-team04-1-/연습장"; OUT = f"{B}/out_yw"
XGB = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42, tree_method="hist", n_jobs=-1)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
df = pd.read_parquet(f"{OUT}/features.parquet"); df = df[df.month >= "2001-01"].copy()
nw = pd.read_csv(f"{B}/out/목표가상향_변수_합치기용.csv", dtype={"code": str})
df = df.merge(nw[["month", "code", "news_missing", "tp_up", "n_earn_up_v1"]], on=["month", "code"], how="left")
df["news_missing"] = df.news_missing.fillna(1); df["tp_up"] = df.tp_up.fillna(0); df["earn_up"] = (df.n_earn_up_v1.fillna(0) > 0).astype(float)
for c in ALL:
    df[c + "_m"] = df[c].isna().astype(float); df[c + "_r"] = df.groupby("month")[c].rank(pct=True).fillna(0.5)
df["y"] = df.groupby("month").target.rank(pct=True)
BASE = [c + "_r" for c in ALL] + [c + "_m" for c in ("beta", "bm", "mom", "ivol", "issue")]
def run(F):
    out = []
    for Y in range(2011, 2022):
        tr = df[(df.month <= f"{Y-1}-11") & df.y.notna()]; te = df[(df.month >= f"{Y}-01") & (df.month <= f"{Y}-12") & df.target.notna()]
        out.append(te[["month", "code", "target", "news_missing"]].assign(score=XGBRegressor(**XGB).fit(tr[F], tr.y).predict(te[F])))
    p = pd.concat(out)
    ic = p.groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
    ic200 = p[p.news_missing == 0].groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
    return ic, ic200
def tt(x): return x.mean() / (x.std() / np.sqrt(len(x)))
res = {}; yrs = {}
for name, F in [("기준(정형 9개)", BASE), ("+목표가 상향", BASE + ["tp_up", "news_missing"]), ("+실적 호재 v1(참고)", BASE + ["earn_up", "news_missing"]), ("+뉴스 미수집 표시만(대조)", BASE + ["news_missing"])]:
    ic, ic200 = run(F); res[name] = (ic, ic200); yrs[name] = ic.groupby(ic.index.str[:4]).mean()
    print(f"{name}: 전체 IC {ic.mean():.4f} (t {tt(ic):.1f}) | 상위200 안 IC {ic200.mean():.4f}", flush=True)
b, b200 = res["기준(정형 9개)"]; rows = []
for name in list(res)[1:]:
    ic, ic200 = res[name]; d = ic - b; d2 = ic200 - b200; win = int((yrs[name] > yrs["기준(정형 9개)"]).sum())
    win16 = int((yrs[name].loc["2016":] > yrs["기준(정형 9개)"].loc["2016":]).sum())
    rows.append({"모델": name, "평균 IC": ic.mean(), "기준 대비": d.mean(), "차이 t": tt(d), "오른 해(/11)": win, "2016–21 오른 해(/6)": win16,
                 "상위200 안 IC 차이": d2.mean(), "상위200 차이 t": tt(d2), "판정": "통과" if (d.mean() > 0 and win >= 6) else "미통과"})
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/tp_fullmodel.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(yrs).to_csv(f"{OUT}/tp_fullmodel_by_year.csv", encoding="utf-8-sig")
pd.set_option("display.width", 250); print(R.round(4).to_string(index=False)); print(pd.DataFrame(yrs).round(4).T.to_string())
