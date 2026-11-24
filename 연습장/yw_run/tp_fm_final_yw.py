"""재현 기준값 확인: 최종 합치기용 파일(전년 말 상위 200, 기사 0건 달 포함)만으로 통제 9개 FM"""
import numpy as np, pandas as pd
O="/home/user/fdi-team04-1-/연습장/out_yw"; ALL=["beta","size","bm","rev","mom","ivol","issue","retail_nb","foreign_nb"]
def tt(x): x=pd.Series(x).dropna(); return x.mean()/(x.std()/np.sqrt(len(x)))
nw=pd.read_csv("/home/user/fdi-team04-1-/연습장/out/목표가상향_변수_합치기용.csv",dtype={"code":str}); nw["code"]=nw.code.str.zfill(6)
f=pd.read_parquet(f"{O}/features.parquet")
u=f.merge(nw[nw.news_missing==0][["month","code","tp_up"]],on=["month","code"])
u=u[u.target.notna()&(u.month>="2001-01")&(u.month<="2021-12")].copy()
for c in ALL: u[c+"_r"]=u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(x):
    B=[]
    for m,g in x.groupby("month"):
        if len(g)<30: continue
        X=np.column_stack([np.ones(len(g))]+[g[c+"_r"] for c in ALL]+[g.tp_up]); B.append(np.linalg.lstsq(X,g.target*100,rcond=None)[0][-1])
    return pd.Series(B)
print("행",len(u),"tp_up=1",int(u.tp_up.sum()))
for p,a,c in [("전체","2001","2021-12"),("2016-21","2016","2021-12")]:
    b=fm(u[(u.month>=a)&(u.month<=c)]); print("FM",p,round(b.mean(),3),"t",round(tt(b),2))
