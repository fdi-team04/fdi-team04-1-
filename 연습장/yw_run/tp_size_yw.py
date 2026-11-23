"""교수님 11/23 제안: 목표가 상향 효과가 작은 기업에서 더 큰가 (상위 200 안에서, 전년 말 시총 기준)
1) 상위 200을 시총 3등분(큰/중간/작은) → 그룹별 통제 9개 FM tp_up 계수
2) FM 상호작용: tp_up × 시총 순위 (음수면 작은 기업일수록 효과 큼)
한계: 뉴스는 상위 200만 수집 → '중소형주'가 아니라 '상위 200 안의 상대적 크기' 비교
"""
import numpy as np, pandas as pd
O="/home/user/fdi-team04-1-/연습장/out_yw"; ALL=["beta","size","bm","rev","mom","ivol","issue","retail_nb","foreign_nb"]
def tt(x): x=pd.Series(x).dropna(); return x.mean()/(x.std()/np.sqrt(len(x)))
nw=pd.read_csv("/home/user/fdi-team04-1-/연습장/out/목표가상향_변수_합치기용.csv",dtype={"code":str})
u=pd.read_parquet(f"{O}/features.parquet").merge(nw[["month","code","tp_up"]],on=["month","code"]); u=u[u.target.notna()].copy()
for c in ALL: u[c+"_r"]=u.groupby("month")[c].rank(pct=True).fillna(0.5)
u["grp"]=u.groupby("month").me.transform(lambda s: pd.qcut(s.rank(method="first"),3,labels=["작은","중간","큰"]))
def fm(x,extra):
    B=[]
    for m,g in x.groupby("month"):
        if len(g)<20 or g.tp_up.sum()<3: continue
        X=np.column_stack([np.ones(len(g))]+[g[c+"_r"] for c in ALL]+[g[e] for e in extra]); B.append(np.linalg.lstsq(X,g.target*100,rcond=None)[0][-len(extra):])
    return pd.DataFrame(B,columns=extra)
rows=[]
for k in ["큰","중간","작은"]:
    for p,a,b in [("전체","2001","2021-12"),("2016-21","2016","2021-12")]:
        B=fm(u[(u.grp==k)&(u.month>=a)&(u.month<=b)],["tp_up"]); rows.append({"그룹":k,"기간":p,"tp_up 효과(%p)":B.tp_up.mean(),"t":tt(B.tp_up),"목표가 기사 비율":u[u.grp==k].tp_up.mean()})
u["inter"]=u.tp_up*u.size_r
for p,a,b in [("전체","2001","2021-12"),("2016-21","2016","2021-12")]:
    B=fm(u[(u.month>=a)&(u.month<=b)],["tp_up","inter"]); rows.append({"그룹":"상호작용(tp_up×크기)","기간":p,"tp_up 효과(%p)":B.inter.mean(),"t":tt(B.inter),"목표가 기사 비율":np.nan})
R=pd.DataFrame(rows); R.to_csv(f"{O}/tp_size.csv",index=False,encoding="utf-8-sig"); print(R.round(3).to_string(index=False))
