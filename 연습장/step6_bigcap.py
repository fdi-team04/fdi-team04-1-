"""강건성: 같은 예측을 대형주 안에서만 평가 (그달 시총 상위 100 / 200 / 나머지)"""
import pandas as pd
OUT = "/home/user/fdi-team04-1-/연습장/out"
rows = []
for s, n in [("0", "0 기준선"), ("1", "1 +통제"), ("2", "2 +보강"), ("3", "3 +수급")]:
    p = pd.read_parquet(f"{OUT}/pred_{s}.parquet")
    p["rk"] = p.groupby("month").me.rank(ascending=False)
    r = {"단계": n}
    for lab, cond in [("전체", p.rk > 0), ("상위100", p.rk <= 100), ("상위200", p.rk <= 200), ("200위 밖", p.rk > 200)]:
        ic = p[cond].groupby("month").apply(lambda x: x.score.rank().corr(x.target.rank()), include_groups=False)
        r[lab] = ic.mean()
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(f"{OUT}/ic_by_size_group.csv", index=False, encoding="utf-8-sig")
print(T.round(4).to_string(index=False))
