"""월별 마지막 거래일을 한 번 받아 tables/last_trading_days.csv로 저장한다.   담당: B-2

03의 월 배정(assign_month)이 이 파일을 읽는다. 평일로 근사하면 12/31 휴장일이나 연휴가 낀 월말에서
실제 마지막 거래일 기사가 당월에 남으므로(계획서 4절 3단계 위반) 실제 거래일을 쓴다.

pykrx와 인터넷이 필요해서 Docker 밖(로컬·Colab)에서 돌린다. Quant 폴더에서 한 번만:
    pip install pykrx
    python src/make_trading_days.py
"""
import time

import pandas as pd
import yaml
from pykrx import stock

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
first = int(cfg["periods"]["warmup"][0][:4])  # 2000
last = int(cfg["periods"]["test"][1][:4])     # 2025

days = []
for year in range(first, last + 1):
    days += list(stock.get_index_ohlcv(f"{year}0101", f"{year}1231", "1001").index)  # 코스피 지수가 있는 날 = 거래일
    time.sleep(1)  # KRX 서버에 연달아 요청하지 않는다

days = pd.Series(pd.to_datetime(days))
month_end = days.groupby(days.dt.to_period("M")).max()  # 월별 마지막 거래일
assert len(month_end) == (last - first + 1) * 12, f"달 수가 맞지 않는다: {len(month_end)}"

path = f"{cfg['tables_dir']}/last_trading_days.csv"
month_end.dt.strftime("%Y-%m-%d").to_frame("date").to_csv(path, index=False)
print(f"{path} 저장: {len(month_end)}개월, {month_end.iloc[0].date()} ~ {month_end.iloc[-1].date()}")
