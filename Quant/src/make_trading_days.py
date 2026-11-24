"""월별 마지막 거래일을 한 번 받아 tables/last_trading_days.csv로 저장한다.   담당: B-2

03의 월 배정(assign_month)이 이 파일을 읽는다. 평일로 근사하면 12/31 휴장일이나 연휴가 낀 월말에서
실제 마지막 거래일 기사가 당월에 남으므로(계획서 4절 3단계 위반) 실제 거래일을 쓴다.

한줄요약 : 거래일 기준 월별 마지막 날을 계산해서 CSV로 만들어주는 코드

pykrx와 인터넷이 필요해서 Docker 밖(로컬·Colab)에서 돌린다. Quant 폴더에서 한 번만:
    pip install pykrx
    python src/make_trading_days.py
"""
import time                     # 서버 요청 간격을 두기 위해 sleep()를 쓴다.

import pandas as pd             # 날짜 처리와 CSV 저장에 사용한다.
import yaml                     # config.yaml 설정 파일을 읽어온다.
from pykrx import stock         # 코스피/주식/지수 데이터 조회용 라이브러리

# config.yaml에서 기간 시작/끝 연도와 테이블 저장 경로를 읽어온다.
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
first = int(cfg["periods"]["warmup"][0][:4])                    # 시작 연도: 예) "2000-01-01" -> 2000
last = int(cfg["periods"]["test"][1][:4])                       # 끝 연도: 예) "2025-12-31" -> 2025

# 각 연도마다 코스피 지수 데이터가 있는 날 = 거래일이라고 보고 그 날짜들을 모두 모은다.
days = []
for year in range(first, last + 1):
    days += list(stock.get_index_ohlcv(f"{year}0101", f"{year}1231", "1001").index)  # 코스피 지수가 있는 날 = 거래일
    time.sleep(1)  # KRX 서버에 연달아 요청하지 않는다.

# 리스트를 datetime 형식으로 바꾸고, 각 월별 마지막 거래일만 남긴다.
days = pd.Series(pd.to_datetime(days))
month_end = days.groupby(days.dt.to_period("M")).max()  # 월별 마지막 거래일

# 총 월 수가 기간에 맞는지 검증한다. 데이터 누락/오류가 있으면 바로 실패하게 만든다.
assert len(month_end) == (last - first + 1) * 12, f"달 수가 맞지 않는다: {len(month_end)}"

# CSV 경로 만들고, 날짜만 YYYY-MM-DD 형식으로 저장한다.
path = f"{cfg['tables_dir']}/last_trading_days.csv"
month_end.dt.strftime("%Y-%m-%d").to_frame("date").to_csv(path, index=False)

# 결과 확인용 출력: 저장된 개월 수와 첫/마지막 거래일을 보여준다.
print(f"{path} 저장: {len(month_end)}개월, {month_end.iloc[0].date()} ~ {month_end.iloc[-1].date()}")
