"""[03] 텍스트 → 종목·월 점수 → 정형 패널 결합 → 순위 변환   담당: B-1(앞부분) · B-2(점수·결합)

입력  data/raw/news/**/*.xlsx                 (B-1 수집 빅카인즈 엑셀, 비공개)
      data/processed/company_list.csv        (회사명 사전)
      data/processed/features.parquet        (A: month, code, 정형 변수, target)
      tables/last_trading_days.csv           (월별 마지막 거래일. src/make_trading_days.py로 한 번 만든다)
      tables/news_targets.csv                (선택. 수집 기록: code, start_month, end_month)

출력  data/processed/articles_clean.parquet  (정제·월 배정한 기사, 원문 포함 → 비공개)
      data/processed/finbert_cache.parquet   (텍스트 해시 → 확률 3개 캐시, 원문 없음)
      tables/news_monthly.csv                (종목·월 뉴스 표, 원문 없음 → 저장소에 올린다)
      data/processed/model_input.parquet     (04_model 입력)

기사가 있는 환경(담당자·Colab)에서는 기사 → 종목·월 뉴스 표를 새로 만들어 저장하고,
기사가 없는 환경(팀원 Docker·채점)에서는 저장소의 종목·월 뉴스 표를 읽어 결합·순위 변환만 한다.

B-1(정아영): load_news ~ classify_type (적재·정제·월 배정·유형 분류, 키워드는 news_rules.py)
B-2(하이안): finbert_probs 이후 (점수 계산·종목·월 뉴스 표·패널 결합·순위 변환)
이용 조건: 빅카인즈 기사 원문은 이용약관상 재배포 제한 → 저장소·제출물에 올리지 않는다.
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""
import functools
import glob
import hashlib
import os
import sys
import time

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # run_all.py로 돌려도 src/의 news_rules를 찾게

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"
TABLES = cfg["tables_dir"]                 # 저장소에 올리는 작은 표 (원문 없음)
NEWS_DIR = f"{cfg['data_dir']}/raw/news"   # 빅카인즈 엑셀 (비공개)
CLEAN = f"{PROC}/articles_clean.parquet"   # 정제·월 배정한 기사 (원문 포함 → 비공개)
OUTLETS = {"머니투데이", "매일경제", "파이낸셜뉴스", "서울경제", "한국경제"}  # 전 기간 기사가 있는 경제지 5곳
LABELS = ["negative", "neutral", "positive"]  # KR-FinBert-SC 출력 라벨
CHUNK = 2000                               # 점수를 이만큼 계산할 때마다 캐시를 저장한다

"""[H3] 뉴스 감성 (+, 대립 −): 뉴스 수집·정제 부분

담당: 정아영(B1). 점수 산출(score_sentiment, news_sentiment)은 하이안(B2)이 이어서 작성한다.

이론: 기업 관련 기사에 담긴 정보는 이후 실적을 예측하지만 주가가 즉시 다 반영하지 못한다
      (Tetlock, Saar-Tsechansky & Macskassy, 2008). 대립 가설은 비관적 논조 뒤 되돌림(Tetlock, 2007).
자료: D5 빅카인즈 경제지 5곳(머니투데이, 매일경제, 파이낸셜뉴스, 서울경제, 한국경제), 연 단위 엑셀.
이용 조건: 기사 원문은 이용약관상 재배포 제한 → 저장소와 제출물에 올리지 않는다(data/raw는 .gitignore).
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""


def load_news(raw_dir):
    """[H3] 뉴스 원자료 적재 (+, 대립 −)

    이론: 위 모듈 설명 참조.
    입력: raw_dir — data/raw/news/ 아래 빅카인즈 엑셀(담당별 하위 폴더까지 읽는다).
    출력: DataFrame(nid, date, outlet, title, body, excluded_flag, source_file)
    규칙: 일자는 YYYYMMDD(시각 없음). 본문은 200자 이내만 제공된다.
          nid = 뉴스 식별자. 숫자로 읽으면 뒷자리가 잘리므로 문자열로 읽는다.
    시점: t월 말까지 공개된 기사만 이후 단계에서 쓴다(여기서는 적재만).
    """
    import warnings

    cols = {"뉴스 식별자": "nid", "일자": "date", "언론사": "outlet", "제목": "title",
            "본문": "body", "분석제외 여부": "excluded_flag"}
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl 스타일 경고(빅카인즈 파일 형식) 무시
        for f in sorted(glob.glob(os.path.join(raw_dir, "**", "*.xlsx"), recursive=True)):
            d = pd.read_excel(f, usecols=lambda c: c in cols, dtype={"뉴스 식별자": str})
            d = d.rename(columns=cols)
            for c in cols.values():
                if c not in d.columns:
                    d[c] = pd.NA
            d["source_file"] = os.path.relpath(f, raw_dir)
            parts.append(d)
    if not parts:
        raise FileNotFoundError(f"{raw_dir}에 xlsx 파일이 없다")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    return df[["nid", "date", "outlet", "title", "body", "excluded_flag", "source_file"]]


def load_name_dict(path="data/processed/company_list.csv"):
    """company_list.csv → {종목코드: [(회사명, first_month, last_month), ...]}
    열 이름은 code/name/first_month/last_month 를 기본으로 하고, 비슷한 이름도 받는다.
    first_month, last_month 는 'YYYY-MM' 문자열, 비어 있으면 제한 없음."""
    cl = pd.read_csv(path, dtype=str, encoding="utf-8")
    pick = lambda *c: next(x for x in c if x in cl.columns)
    ccode = pick("code", "stock_code", "종목코드")
    cname = pick("name", "names", "company_name", "회사명")
    cfirst = pick("first_month", "first", "시작월")
    clast = pick("last_month", "last", "종료월")
    nd = {}
    for _, r in cl.iterrows():
        code = str(r[ccode]).zfill(6)  # 종목코드는 문자열 6자리
        nd.setdefault(code, []).append(
            (r[cname], r[cfirst] if pd.notna(r[cfirst]) else None,
             r[clast] if pd.notna(r[clast]) else None))
    return nd

def clean_news(df, name_dict):
    """[H3] 분석제외·중복 제거와 제목 기준 종목 매칭 (+, 대립 −)

    입력: df — load_news 출력.
          name_dict — {종목코드: [(회사명, first_month, last_month), ...]}  (load_name_dict 출력).
                      예전 형식 {종목코드: [회사명, ...]} 도 받는다(기간 제한 없음).
    출력: DataFrame(code, date, title, body)
    규칙: ⓪ 경제지 5곳(OUTLETS) 기사만 남김
          ① 분석제외 표시('중복', '예외')가 있는 기사 제거
          ② 뉴스 식별자가 같은 기사, (제목, 언론사, 일자)가 같은 기사 제거
          ③ 제목에 회사명이 있는 기사만 남김
             - 회사명 앞뒤에 영문·숫자가 바로 붙으면 다른 단어로 본다(SK → SKC 제외)
             - 더 긴 다른 회사명 안에 들어 있는 이름은 그 긴 이름을 지운 뒤 찾는다(SK → SK가스 제외)
             - 기사 날짜의 달이 회사명의 first_month~last_month 안일 때만 그 종목에 붙인다
             - 같은 이름이 같은 달에 두 종목 이상에 해당하면 모호 기사로 보고 뺀다
          ④ 기관 열에만 회사명이 있는 기사는 보류(사용하지 않음)
          ⑤ 한 종목의 이름 두 개가 한 제목에 있어 같은 기사가 두 번 붙으면 한 번만 남김
    기사 수는 신뢰도 점검용으로만 기록하고 예측 변수로 쓰지 않는다.
    """
    import re

    d = df.copy()
    d["title"] = d["title"].fillna("").astype(str).str.replace("車", "차")
    d["body"] = d["body"].fillna("").astype(str).str.replace("車", "차")
    # ⓪ 경제지 5곳 기사만 남김 (언론사 구성이 바뀌면 점수가 흔들린다)
    d = d[d["outlet"].isin(OUTLETS)]
    # ① 분석제외 표시('중복', '예외')가 있는 기사 제거
    flag = d["excluded_flag"].fillna("").astype(str)
    d = d[~flag.str.contains("중복|예외")]
    # ② 뉴스 식별자가 같은 기사(여러 검색 파일에 겹쳐 받은 것), (제목, 언론사, 일자)가 같은 기사 제거
    d = d.dropna(subset=["date"])
    d = d[~(d["nid"].notna() & d.duplicated("nid"))]
    d = d.drop_duplicates(["title", "outlet", "date"])
    d = d.reset_index(drop=True)
    d["_ym"] = d["date"].dt.strftime("%Y-%m")

    # 이름 → [(종목코드, first, last)] 로 뒤집는다 (같은 이름이 여러 종목에 쓰인 경우 대비)
    by_name = {}
    for code, items in name_dict.items():
        for it in items:
            if isinstance(it, str):
                it = (it, None, None)
            n, first, last = it
            if not isinstance(n, str):
                continue
            n = n.strip().replace("車", "차")
            if len(n) < 2:
                continue
            by_name.setdefault(n, []).append((code, first, last))
    all_names = sorted(by_name, key=len, reverse=True)

    # ③ 제목에 회사명이 있는 기사만 남김 (경계 + 긴 이름 우선 + 기간)
    edge_l, edge_r = r"(?<![0-9A-Za-z])", r"(?![0-9A-Za-z])"
    out, n_ambiguous = [], 0
    for n, recs in by_name.items():
        # n 을 포함하는 더 긴 다른 회사명 (SK → SKC, SK가스 ...)
        longer = [m for m in all_names if len(m) > len(n) and n in m]
        titles = d["title"]
        if longer:
            mask_pat = "|".join(re.escape(m) for m in longer)  # 긴 이름부터 지운다
            titles = titles.str.replace(mask_pat, " ", regex=True)
        pat = edge_l + re.escape(n) + edge_r
        hit = d[titles.str.contains(pat, regex=True)]
        if hit.empty:
            continue
        # 기사 날짜가 해당 종목의 기간 안인지 확인
        ok = pd.DataFrame(index=hit.index)
        for i, (code, first, last) in enumerate(recs):
            m = pd.Series(True, index=hit.index)
            if isinstance(first, str) and first:
                m &= hit["_ym"] >= first
            if isinstance(last, str) and last:
                m &= hit["_ym"] <= last
            ok[i] = m
        cnt = ok.sum(axis=1)
        n_ambiguous += int((cnt > 1).sum())          # 같은 이름이 두 종목에 겹침 → 뺀다
        use = ok[cnt == 1]
        for i, (code, first, last) in enumerate(recs):
            rows = use.index[use[i]]
            if len(rows):
                out.append(hit.loc[rows].assign(code=code))
    print(f"[clean_news] 같은 이름이 두 종목에 겹쳐 뺀 기사: {n_ambiguous}건")

    cols = ["code", "date", "title", "body"]
    if not out:
        return pd.DataFrame(columns=cols)
    # ⑤ 같은 기사가 한 종목에 두 번 붙은 것은 한 번만 남김
    res = pd.concat(out, ignore_index=True).drop_duplicates(["code", "title", "outlet", "date"])
    print(f"[clean_news] 읽은 기사 {len(df)}건 → 정리 후 {len(d)}건 → 종목에 붙인 기사 {len(res)}건")
    return res[cols]


def assign_month(df, trading_days):
    """[H3] 월 귀속 (+, 대립 −)

    입력: df — clean_news 출력.
          trading_days — 거래일 목록(월별 마지막 거래일 판별용).
    출력: DataFrame(code, month, date, title, body)
    규칙: t월 마지막 거래일 전날까지의 기사는 t월, 마지막 거래일 이후 기사는 t+1월로 보낸다.
    시점: 시각이 없으므로 보수적으로 마지막 거래일 당일 기사는 t+1월로 보낸다(미래정보 차단).
    기사가 없는 달은 결측으로 두고 2.3절 규칙(월별 중앙값 + 결측 표시 열)을 따른다.
    """
    td = pd.to_datetime(pd.Series(list(trading_days)))
    last_td = td.groupby(td.dt.to_period("M")).max()  # 월별 마지막 거래일
    
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    per = d["date"].dt.to_period("M")
    cut = per.map(last_td)
    if cut.isna().any():
        bad = sorted(per[cut.isna()].astype(str).unique())
        raise ValueError(f"trading_days에 없는 달의 기사가 있다: {bad}")
    # 마지막 거래일 당일과 그 이후 기사는 다음 달(t+1)로 보낸다
    month = per.where(d["date"] < cut, per + 1)
    d["month"] = month.astype(str)
    return d[["code", "month", "date", "title", "body"]]


def classify_type(df):
    """[H3] 기사 유형 분류 (B-1 정아영)

    입력: df — assign_month 출력.
    출력: df + news_type 열(광고, 공시형, 논란형, 화제형, 기타).
    규칙: 키워드와 우선순위는 src/news_rules.py 한 곳에서 관리한다(키워드 규칙 v1 초안).
          시황·리포트는 기타로 먼저 거르고 → 광고 → 공시형 → 논란형 → 화제형 → 기타.
          공시형과 논란형이 같이 걸리면 논란형. 제목에서 먼저 찾고 없을 때만 본문 200자에서 찾는다.
          분석에는 화제형만 쓴다(aggregate_monthly).
    """
    from news_rules import classify

    out = df.copy()
    out["news_type"] = [classify(t, b)[0] for t, b in zip(out["title"], out["body"])]
    return out


# ---- 아래는 B2(하이안) 담당: 점수 계산 이후 ----


@functools.lru_cache(maxsize=1)
def _finbert():
    """KR-FinBert-SC를 한 번만 불러온다 (GPU가 있으면 GPU, 없으면 CPU)."""
    import torch
    from transformers import pipeline
    clf = pipeline("text-classification", model=cfg["text_model"],
                   revision=cfg.get("text_model_rev"), top_k=None,
                   truncation=True,  # 토큰 제한 512 안전장치
                   device="cuda" if torch.cuda.is_available() else "cpu")
    rev = getattr(clf.model.config, "_commit_hash", None)
    print(f"[finbert] 장치 {clf.device}, 모델 버전(커밋) {rev} → config.yaml의 text_model_rev에 적어 고정한다")
    return clf


def finbert_probs(texts: pd.Series) -> pd.DataFrame:
    """[H3] KR-FinBert-SC 라벨별 확률 (+, 대립 −)

    입력: texts — 기사 텍스트(제목 + 본문).
    출력: DataFrame(negative, neutral, positive), texts와 같은 인덱스
    규칙: 사전학습 모델을 그대로 쓰고 재학습하지 않는다(이용 조건).
          config.yaml에 text_model_rev(커밋 해시)를 적으면 그 버전으로 고정된다.
    """
    out = _finbert()(texts.tolist(), batch_size=32)  # 32개씩 묶어 계산
    probs = pd.DataFrame([{d["label"]: d["score"] for d in o} for o in out],
                         index=texts.index)
    return probs[LABELS]


def score_articles(articles: pd.DataFrame, backend: str = "finbert") -> pd.DataFrame:
    """[H3] 기사별 감성 점수 (+, 대립 −)

    입력: articles — B-1 가공 빅카인즈(code, month, date, title, body[, news_type]).
          backend — "finbert"(config의 text_model). "llm"(upstage ai)은 나중에 추가.
    출력: DataFrame(code, month, date[, news_type], p_neg, p_neu, p_pos, s), s = P(긍정) − P(부정) ∈ [−1, 1]
    규칙: 제목 + 본문(200자)을 넣는다. 모델 버전 고정, 재학습 없음.
          캐시(finbert_cache.parquet)에는 텍스트 해시와 확률 3개만 둔다(원문 없음).
          캐시에 없는 새 텍스트만 계산하고 CHUNK건마다 저장하므로, 기사를 더 받거나 중간에 끊겨도 이어서 돈다.
          계산한 건수와 걸린 시간을 출력한다(계획서 7절 시간 측정).
    """
    if backend != "finbert":
        raise NotImplementedError("LLM(Upstage) 백엔드는 나중에 추가")

    articles = articles.reset_index(drop=True)
    text = articles["title"].fillna("") + " " + articles["body"].fillna("")
    key = text.map(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())  # 원문 대신 해시를 캐시 키로

    cache = f"{PROC}/finbert_cache.parquet"
    parts = [pd.read_parquet(cache)] if os.path.exists(cache) else []
    todo = text[~key.isin(parts[0]["key"])] if parts else text
    todo = todo.drop_duplicates()  # 캐시에 없는 새 텍스트만, 같은 텍스트는 한 번만
    print(f"[score_articles] 기사 {len(text)}건 중 새로 계산할 텍스트 {len(todo)}건")
    if len(todo):
        _finbert()  # 모델을 먼저 불러 둔다 (불러오는 시간은 측정에서 뺀다)
    t0 = time.perf_counter()
    for i in range(0, len(todo), CHUNK):
        part = todo.iloc[i:i + CHUNK]
        parts.append(finbert_probs(part).assign(key=key[part.index]))
        pd.concat(parts, ignore_index=True).to_parquet(cache, index=False)  # 중간 저장
        n, sec = i + len(part), time.perf_counter() - t0
        print(f"[score_articles] {n}/{len(todo)}건, {sec:.0f}초 (건당 {sec / n:.3f}초)")

    probs = pd.concat(parts).set_index("key") if parts else pd.DataFrame(columns=LABELS, dtype=float)
    p = probs.loc[key]  # 기사 순서대로 확률을 붙인다
    out = articles.drop(columns=["title", "body"])
    out["p_neg"], out["p_neu"], out["p_pos"] = p["negative"].values, p["neutral"].values, p["positive"].values
    out["s"] = out["p_pos"] - out["p_neg"]
    return out


def load_targets(path):
    """수집 기록(code, start_month, end_month) → 수집 대상 (month, code) 전체. 파일이 없으면 None.

    수집 기록은 B-1이 tables/news_targets.csv에 적는다(다른 열은 자유롭게 추가).
    이 목록이 있으면 종목·월 뉴스 표에 "수집했지만 화제형 기사가 없는 칸"(no_news = 1)까지 들어간다.
    """
    if not os.path.exists(path):
        return None
    t = pd.read_csv(path, dtype=str)
    rows = [(str(m), code.zfill(6))
            for code, first, last in zip(t["code"], t["start_month"], t["end_month"])
            for m in pd.period_range(first, last, freq="M")]
    return pd.DataFrame(rows, columns=["month", "code"]).drop_duplicates()


def aggregate_monthly(scored: pd.DataFrame, targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """[H3] 종목·월 뉴스 표 (+, 대립 −)   계획서 9절의 종목·월 표

    입력: scored — score_articles 출력(news_type 열 포함).
          targets — 수집 대상 (month, code) 목록(load_targets 출력). 없으면 None.
    출력: DataFrame(month, code, topic_tone, n_topic, no_news)
    규칙: 화제형(news_type == "화제형") 기사만 쓴다.
          topic_tone = 그 달 화제형 기사 s의 평균, n_topic = 화제형 기사 수(점검용, 모델에 넣지 않음).
          no_news = 화제형 기사가 없으면 1. targets가 있으면 수집 대상 칸이 전부 행이 되고,
          없으면 화제형 기사가 있는 칸만 행이 된다(빠진 칸은 결합할 때 no_news = 1이 된다).
    """
    topic = scored[scored["news_type"] == "화제형"]
    text = (topic.groupby(["month", "code"], as_index=False)  # 같은 달, 종목 기사끼리 묶음
                 .agg(topic_tone=("s", "mean"), n_topic=("s", "size")))
    if targets is not None:
        n_cells = len(text)
        text = targets.merge(text, how="left", on=["month", "code"])
        print(f"[aggregate_monthly] 수집 기록 밖이라 뺀 칸: {n_cells - text['n_topic'].notna().sum()}개")
    text["n_topic"] = text["n_topic"].fillna(0).astype(int)
    text["no_news"] = (text["n_topic"] == 0).astype(int)
    return text


def merge_text_features(panel: pd.DataFrame, text: pd.DataFrame | None) -> pd.DataFrame:
    """[H3] 정형 패널에 텍스트 변수 결합 (+, 대립 −)

    입력: panel — features.parquet(month, code, 정형 변수, target).
          text — 종목·월 뉴스 표(aggregate_monthly 출력 = tables/news_monthly.csv). 표가 없으면 None.
    출력: panel + (topic_tone, n_topic, no_news)
    규칙: left join (키: month, code / validate="one_to_one"), 결합 전후 행 수가 같은지 확인.
          no_news = 그 달 화제형 기사가 없으면 1 (표에 없는 수집 대상 밖 종목 포함). 이때 n_topic = 0,
          topic_tone은 결측으로 둔다. 표가 아예 없으면 모든 행이 no_news = 1이다(뉴스 없이 통과).
    """
    if text is None:
        return panel.assign(topic_tone=float("nan"), n_topic=0, no_news=1)
    out = panel.merge(text, how="left", on=["month", "code"], validate="one_to_one")
    assert len(out) == len(panel), "결합 후 행 수가 바뀜"
    out["topic_tone"] = out["topic_tone"].astype(float)     # 표가 비어 있어도 숫자 열로
    out["no_news"] = out["no_news"].fillna(1).astype(int)   # 표에 없는 종목·월도 기사 없음
    out["n_topic"] = out["n_topic"].fillna(0).astype(int)   # 빈칸 정리
    return out


def rank_transform(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """월별 횡단면 순위 변환 (2.3절 규칙)

    입력: panel — merge_text_features 출력. cols — 변환할 변수 목록(target·n_topic·no_news는 제외해 원값 유지).
    출력: panel(cols를 0~1 순위 백분위로 교체) + 변수명_missing(0/1)
    규칙: 순위 백분위 = rank / (n + 1) (n = 그 달 값이 있는 종목 수, 동순위는 평균 순위). 정의서 공통 규칙.
          결측 = 0.5(중앙값) + 변수명_missing. 결측이 없는 변수도 표시 열을 만들어 기간별 열 구성을 같게 한다.
    시점: 그 달 단면만 쓰므로 기간을 넘는 통계 추정이 없다(test 누수 방지).
    """
    miss = panel[cols].isna().astype(int).add_suffix("_missing")  # 결측 표시 열
    g = panel.groupby("month")[cols]  # 월별로 묶어 순위 변환
    pct = g.rank() / (g.transform("count") + 1)  # 동순위 평균, 결측은 제외
    out = panel.copy()
    out[cols] = pct.fillna(0.5)  # 결측은 중앙값 0.5로 채움
    return out.join(miss)


def build_news_table() -> pd.DataFrame:
    """[H3] 비공개 기사 → 종목·월 뉴스 표 (기사가 있는 환경에서만 부른다)

    순서: (엑셀 적재 → 정제 → 월 배정) → 유형 분류 → 화제형 점수 → 종목·월 집계.
    괄호 안은 오래 걸려 articles_clean.parquet에 저장해 다시 쓴다. 엑셀을 새로 받았으면 그 파일을 지우고 돌린다.
    유형 분류는 매번 새로 하므로 news_rules.py를 고치면 바로 반영되고, 점수는 새 텍스트만 계산된다.
    """
    if os.path.exists(CLEAN):
        arts = pd.read_parquet(CLEAN)
        print(f"[03] articles_clean.parquet 다시 사용: {len(arts)}건 (엑셀을 새로 받았으면 이 파일을 지우고 실행)")
    else:
        days_path = f"{TABLES}/last_trading_days.csv"
        if not os.path.exists(days_path):
            raise FileNotFoundError(f"{days_path}가 없다 → `python src/make_trading_days.py`로 한 번 만든다")
        arts = clean_news(load_news(NEWS_DIR), load_name_dict(f"{PROC}/company_list.csv"))  # 적재 → 종목코드 붙이기
        arts = assign_month(arts, pd.read_csv(days_path)["date"])                           # 실제 마지막 거래일 기준
        arts.to_parquet(CLEAN, index=False)

    arts = classify_type(arts)
    print("[03] 유형별 기사 수:", arts["news_type"].value_counts().to_dict())
    scored = score_articles(arts[arts["news_type"] == "화제형"])  # 계획서 7절: 화제형에만 점수를 매긴다
    return aggregate_monthly(scored, load_targets(f"{TABLES}/news_targets.csv"))


if __name__ == "__main__":
    # 1) 종목·월 뉴스 표: 기사(비공개)가 있으면 새로 만들어 저장하고, 모델은 항상 저장된 표를 읽는다
    table_path = f"{TABLES}/news_monthly.csv"
    if os.path.exists(CLEAN) or glob.glob(f"{NEWS_DIR}/**/*.xlsx", recursive=True):
        os.makedirs(TABLES, exist_ok=True)
        build_news_table().to_csv(table_path, index=False)
    if os.path.exists(table_path):
        text = pd.read_csv(table_path, dtype={"month": str, "code": str})
        print(f"[03] 종목·월 뉴스 표 {len(text)}칸, 그중 화제형 기사가 있는 칸 {(text['no_news'] == 0).sum()}칸")
    else:
        text = None  # 뉴스가 막힌 경우: 뉴스 열을 비운 채 통과한다 (계획서 2절, 수급 단계까지의 모델)
        print("[03] 종목·월 뉴스 표가 없다 → 뉴스 열을 비운 채 진행")

    # 2) 정형 패널에 붙이고 순위 변환 → 04 입력
    feat_path = f"{PROC}/features.parquet"
    if not os.path.exists(feat_path):  # 점수만 먼저 낼 때: 종목·월 뉴스 표까지는 끝난 상태
        raise SystemExit(f"[03] {feat_path}가 없어 여기서 멈춘다 → 02를 돌린 뒤 다시 실행한다")
    df = merge_text_features(pd.read_parquet(feat_path), text)
    keep = {"target", "n_topic", "no_news"}  # 숫자지만 원값 유지
    cols = [c for c in df.select_dtypes("number").columns if c not in keep]
    df = rank_transform(df, cols).drop(columns="topic_tone_missing")  # no_news와 같은 열이라 뺀다
    df.to_parquet(f"{PROC}/model_input.parquet", index=False)
    print(f"[03] model_input {df.shape}, 화제형 기사가 있는 행 {1 - df['no_news'].mean():.1%}")
