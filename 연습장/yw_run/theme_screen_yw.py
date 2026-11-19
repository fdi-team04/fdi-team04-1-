"""테마 빠른 스크리닝: theme_monthly.csv(영섭 PC 결과) → 테마마다 같은 기준으로 자동 판정
판정 기준 (결과 보기 전에 정함, 11/19):
  0) 데이터: 테마 기사가 있는 종목-월 300개 이상, 2016-21에도 30개 이상
  1) 단순 차이: 다음 달 수익률 (있음-없음) 이 기대 방향이고 |t| >= 2
  2) 통제 후: 다른 변수 9개 넣은 FM에서 기대 방향이고 |t| >= 2   ← 테마를 여러 개 보니 우연 통과를 막는 핵심 관문
  3) 최근 유지: 2016-21 구간 FM 계수가 기대 방향 (반대로 뒤집히지 않음)
  → 1·2·3 모두 통과: '파 보기'(사람 확인 50건 정밀도 80%, XGBoost IC 11년 중 6년 이상 개선 확인)
  → 2만 통과/1·3 중 하나 실패: '보류'(기간·정의 한 번만 점검)
  → 2 실패: '빼기'(기록하고 종료)
  기대 방향 '?'는 양쪽 검정(|t| >= 2).
쓰는 법: python theme_screen_yw.py <theme_monthly.csv> [결과폴더 이름표]
"""
import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/fdi-team04-1-/연습장/news")
from theme_rules import THEMES
SRC = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else "full"
O = "/home/user/fdi-team04-1-/연습장/out_yw"; P = f"{O}/theme_screen_{TAG}"; os.makedirs(P, exist_ok=True)
ALL = ["beta", "size", "bm", "rev", "mom", "ivol", "issue", "retail_nb", "foreign_nb"]
def tt(x): x = pd.Series(x).dropna(); return x.mean() / (x.std() / np.sqrt(len(x))) if len(x) > 2 else np.nan
nm = pd.read_csv(SRC, dtype={"code": str}); nm["code"] = nm.code.str.zfill(6)
f = pd.read_parquet(f"{O}/features.parquet").sort_values(["code", "month"])
dec = f[f.month.str[5:] == "12"].copy(); dec["rk"] = dec.groupby("month").me.rank(ascending=False)
top = dec[dec.rk <= 200].assign(year=lambda d: (d.month.str[:4].astype(int) + 1).astype(str))[["code", "year"]]
f["year"] = f.month.str[:4]
u = f.merge(top, on=["code", "year"]).merge(nm, on=["month", "code"], how="left")
u["n_all"] = u.n_all.fillna(0)
u = u[(u.groupby(["code", "year"]).n_all.transform("sum") > 0) & u.target.notna() & (u.month >= "2001-01")].copy()
for c in ALL: u[c + "_r"] = u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(d, col):
    B = []
    for m, g in d.groupby("month"):
        if len(g) < 30 or g[col].nunique() < 2: continue
        X = np.column_stack([np.ones(len(g))] + [g[c + "_r"] for c in ALL] + [g[col]]); B.append(np.linalg.lstsq(X, g.target * 100, rcond=None)[0][-1])
    return pd.Series(B)
rows = []
for k, (sgn, _, _) in THEMES.items():
    if k not in u: continue
    u["fl"] = (u[k].fillna(0) > 0).astype(float)
    n_ev = int(u.fl.sum()); n_ev_recent = int(u[u.month >= "2016-01"].fl.sum())
    d = u.groupby(["month", "fl"]).target.mean().unstack() * 100
    dd = (d[1.0] - d[0.0]).dropna() if 1.0 in d else pd.Series(dtype=float)
    d0 = u.groupby(["month", "fl"]).ret.mean().unstack() * 100
    same = (d0[1.0] - d0[0.0]).dropna() if 1.0 in d0 else pd.Series(dtype=float)
    b_all = fm(u, "fl"); b_rec = fm(u[u.month >= "2016-01"], "fl")
    s = {"+": 1, "-": -1, "?": 0}[sgn]
    ok = lambda m, t: (abs(t) >= 2) if s == 0 else (np.sign(m) == s and abs(t) >= 2)
    g0 = n_ev >= 300 and n_ev_recent >= 30
    g1 = ok(dd.mean(), tt(dd)); g2 = ok(b_all.mean(), tt(b_all))
    g3 = True if s == 0 else (np.sign(b_rec.mean()) == s)
    verdict = "데이터 부족" if not g0 else ("파 보기" if (g1 and g2 and g3) else ("보류" if g2 else "빼기"))
    rows.append({"테마": k, "기대": sgn, "사건 종목-월": n_ev, "2016-21 사건": n_ev_recent,
                 "같은달 차이(참고)": same.mean(), "t같은달": tt(same), "다음달 차이(%p)": dd.mean(), "t차이": tt(dd),
                 "FM 계수(%p)": b_all.mean(), "tFM": tt(b_all), "FM 2016-21": b_rec.mean(), "tFM최근": tt(b_rec),
                 "①데이터": g0, "②단순": g1, "③통제": g2, "④최근": g3, "판정": verdict})
T = pd.DataFrame(rows); T.to_csv(f"{P}/theme_screen.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(T.round(2).to_string(index=False))
