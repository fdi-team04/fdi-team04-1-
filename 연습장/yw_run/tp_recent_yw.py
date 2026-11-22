"""목표가 상향(tp_up) 최근 약화 확인: 연도별 다음 달 차이, 상위 200 안 큰/작은 종목, 통제 9개 FM (2001-2021, 판정과 같은 표본)"""
import numpy as np, pandas as pd
O="/home/user/fdi-team04-1-/연습장/out_yw"; ALL=["beta","size","bm","rev","mom","ivol","issue","retail_nb","foreign_nb"]
def tt(x): x=pd.Series(x).dropna(); return x.mean()/(x.std()/np.sqrt(len(x))) if len(x)>2 else np.nan
nm=pd.read_csv("/home/user/fdi-team04-1-/연습장/out/theme_monthly_v1_input.csv",dtype={"code":str}); nm["code"]=nm.code.str.zfill(6)
f=pd.read_parquet(f"{O}/features.parquet")
dec=f[f.month.str[5:]=="12"].copy(); dec["rk"]=dec.groupby("month").me.rank(ascending=False)
top=dec[dec.rk<=200].assign(year=lambda d:(d.month.str[:4].astype(int)+1).astype(str))[["code","year"]]
f["year"]=f.month.str[:4]
u=f.merge(top,on=["code","year"]).merge(nm,on=["month","code"],how="left"); u["n_all"]=u.n_all.fillna(0)
u=u[(u.groupby(["code","year"]).n_all.transform("sum")>0)&u.target.notna()].copy()
u["tp"]=(u.목표가_상향.fillna(0)>0).astype(float)
d=u.groupby(["month","tp"]).target.mean().unstack()*100; dd=(d[1.0]-d[0.0]).dropna()
yr=dd.groupby(dd.index.str[:4]).agg(["mean","count"]); yr["t"]=dd.groupby(dd.index.str[:4]).apply(tt)
print("연도별 다음 달 차이(%p):"); print(yr.round(2).T.to_string())
print("양(+)인 해:", int((yr["mean"]>0).sum()),"/",len(yr), "| 2016-21:", int((yr.loc["2016":"2021","mean"]>0).sum()),"/6")
u["big"]=u.groupby("month").me.rank(pct=True)>0.5
for b,lab in [(True,"상위200 중 큰 절반"),(False,"작은 절반")]:
    x=u[u.big==b]; d2=x.groupby(["month","tp"]).target.mean().unstack()*100; z=(d2[1.0]-d2[0.0]).dropna()
    for p,a,c in [("전체","2001","2021-12"),("2016-21","2016","2021-12")]:
        zz=z[(z.index>=a)&(z.index<=c)]; print(lab,p,round(zz.mean(),2),"t",round(tt(zz),2))
for c in ALL: u[c+"_r"]=u.groupby("month")[c].rank(pct=True).fillna(0.5)
def fm(x):
    B=[]
    for m,g in x.groupby("month"):
        if len(g)<30: continue
        X=np.column_stack([np.ones(len(g))]+[g[c+"_r"] for c in ALL]+[g.tp]); B.append(np.linalg.lstsq(X,g.target*100,rcond=None)[0][-1])
    return pd.Series(B)
for p,a,c in [("2001-10","2001","2010-12"),("2011-15","2011","2015-12"),("2016-21","2016","2021-12"),("전체","2001","2021-12")]:
    b=fm(u[(u.month>=a)&(u.month<=c)]); print("FM",p,round(b.mean(),2),"t",round(tt(b),2))
