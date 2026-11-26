"""OpenDART 재무제표 받기 (팀원 PC에서 실행, 이 저장소 환경에서는 테스트하지 못함)
준비: 1) https://opendart.fss.or.kr 회원가입 → 인증키 신청  2) pip install OpenDartReader pandas
실행: 인증키를 환경변수로 넣고 실행 (키를 코드·깃·톡방에 쓰지 말 것)
  Windows PowerShell:  $env:DART_KEY="발급받은키";  python dart_fetch.py codes.csv
  codes.csv = 종목코드 한 열(code). 예원 정형 데이터의 Stock_Code를 중복 없이 저장하면 됨
주의: DART 재무 API는 2015년 사업연도부터만 있음 → 2015–2021만 생김.
미래 정보 방지: 사업보고서(12월 결산)는 다음 해 3월 말까지 공시 → 4월 말부터 사용(available_month 칸).
"""
import os, sys, time, pandas as pd, OpenDartReader
dart = OpenDartReader(os.environ["DART_KEY"])
codes = pd.read_csv(sys.argv[1], dtype=str).iloc[:, 0].str.zfill(6).unique()
NEED = {"매출액": ["ifrs-full_Revenue", "ifrs_Revenue"], "매출원가": ["ifrs-full_CostOfSales", "ifrs_CostOfSales"],
        "매출총이익": ["ifrs-full_GrossProfit", "ifrs_GrossProfit"], "자산총계": ["ifrs-full_Assets", "ifrs_Assets"],
        "자본총계": ["ifrs-full_Equity", "ifrs_Equity"], "당기순이익": ["ifrs-full_ProfitLoss", "ifrs_ProfitLoss"],
        "영업활동현금흐름": ["ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs_CashFlowsFromUsedInOperatingActivities"]}
rows = []
for i, c in enumerate(codes):
    for y in range(2015, 2022):
        try:
            fs = dart.finstate_all(c, y, reprt_code="11011", fs_div="CFS")
            if fs is None or len(fs) == 0: fs = dart.finstate_all(c, y, reprt_code="11011", fs_div="OFS")
        except Exception as e:
            fs = None
        r = {"code": c, "year": y, "available_month": f"{y+1}-04"}
        if fs is not None and len(fs):
            for k, ids in NEED.items():
                v = fs[fs.account_id.isin(ids)].thstrm_amount
                r[k] = pd.to_numeric(v.str.replace(",", ""), errors="coerce").dropna().iloc[0] if len(v) else None
        rows.append(r); time.sleep(0.15)
    if i % 50 == 0: print(i, len(codes), flush=True); pd.DataFrame(rows).to_csv("dart_finstate.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows).to_csv("dart_finstate.csv", index=False, encoding="utf-8-sig"); print("저장: dart_finstate.csv")
