"""아영 run_clean3.py(=Quant/src/03_text_features.py 의 clean_news)에 끼워 넣는 보정 2가지 — 팀 원본 코드는 고치지 않음

문제 1. company_list_최종.csv 를 4열(first_month, last_month)로만 바꾸면 '이름을 쓴 기간'(name_first, name_last)이 사라짐
        → 이름이 겹치는 회사(SK, 삼성물산, 오리온, 한국타이어, 만도, 한진해운 …)는 기사가 '모호'로 빠지거나 엉뚱한 종목에 붙음
        → 해결: to_name_dict() 로 이름마다 (이름, 시작, 끝) 을 만들어 clean_news 에 그대로 넘김
           시작/끝 = 그 이름 행의 name_first/name_last, 비어 있으면 그 종목 상장기간(first_month/last_month)
문제 2. clean_news 는 영문·숫자 경계만 봄 → 'SK텔'→SK, 'CJ오쇼핑'→CJ, 'KCC건설'→KCC, '기술대상'→대상, '…-SK증권' 리포트 꼬리표 → SK
        → 해결: postfilter() 로 clean_news 결과 중 match_rules v2 가 '그 회사 기사'라고 인정한 것만 남김

쓰는 법 (run_clean3.py 안에서):
    from 아영_정리보정 import to_name_dict, postfilter
    name_dict = to_name_dict("company_list_최종.csv")        # load_name_dict(...) 대신
    clean = clean_news(raw, name_dict)
    clean = postfilter(clean, "company_list_최종.csv")        # 추가 한 줄
"""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from match_rules import build_index, match_title


def to_name_dict(path):
    c = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("name_first", "name_last"):
        if col not in c: c[col] = ""
    c["lf"] = c.groupby("code").first_month.transform("max")
    c["ll"] = c.groupby("code").last_month.transform("max")
    nd = {}
    for r in c.itertuples():
        nd.setdefault(r.code, []).append((r.name, r.name_first or r.lf or None, r.name_last or r.ll or None))
    return nd


def postfilter(clean, path):
    c = pd.read_csv(path, dtype=str, keep_default_na=False)
    idx = build_index(c); ns = sorted(idx, key=len, reverse=True)
    ym = pd.to_datetime(clean["date"]).dt.strftime("%Y-%m")
    keep = [code in match_title(t, y, idx, ns) for t, y, code in zip(clean["title"].fillna(""), ym, clean["code"])]
    out = clean[keep].copy()
    print(f"[postfilter] {len(clean):,} → {len(out):,} (잘못 붙은 {len(clean) - len(out):,}건 제거)")
    return out
