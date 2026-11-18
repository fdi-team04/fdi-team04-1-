"""감성 FM 기간별 (sent_fm_by_period.csv): 통제 9개 + 논조 지표, 2002-10 / 2011-15 / 2016-21"""
import re,sys
src=open('/home/user/fdi-team04-1-/연습장/yw_run/sent_analysis_yw.py').read()
src=src.split("# XGBoost 3단계")[0]
exec(src)
out=[]
for sv in ["s_all","s_noprice","surprise"]:
    u["sv_r"]=u.groupby("month")[sv].rank(pct=True).fillna(0.5)
    cols=[c+"_r" for c in ALL]+["sv_r","n_r","has"]
    for a,b in [("2002-04","2010-12"),("2011-01","2015-12"),("2016-01","2021-12"),("2011-01","2021-12"),("2002-04","2021-12")]:
        Bs=[]
        for m,g in u[(u.month>=a)&(u.month<=b)].groupby("month"):
            X=np.column_stack([np.ones(len(g))]+[g[c] for c in cols]); Bs.append(np.linalg.lstsq(X,g.target*100,rcond=None)[0][1:])
        Bs=pd.DataFrame(Bs,columns=cols); out.append({"변수":sv,"기간":f"{a[:4]}-{b[2:4]}","계수":Bs.sv_r.mean(),"t":tt(Bs.sv_r)})
O2=pd.DataFrame(out); print(O2.round(3).to_string(index=False)); O2.to_csv(f"{P}/sent_fm_by_period.csv",index=False,encoding="utf-8-sig")
# 기간별 5분위 다음달
for a,b in [("2001-01","2010-12"),("2011-01","2015-12"),("2016-01","2021-12")]:
    z=Q.loc[a:b]["target"]; d=(z[5]-z[1]).dropna(); print(a[:4],b[:4],"Q5-Q1 다음달",round(d.mean(),3),round(tt(d),2))
