"""[03] 텍스트 → 종목·월 점수 → 정형 패널 결합 → 순위 변환   담당: B-1(앞부분) · B-2(점수·결합)

입력  data/processed/articles_clean.csv      (B-1 가공 빅카인즈: code, month, date, title, body)
      data/processed/features.parquet        (A: month, code, 정형 변수, target)

출력  data/processed/article_scores.parquet  (기사별 점수 캐시, 원문 없음)
      data/processed/model_input.parquet     (04_model 입력)

B-1(정아영): load_news, clean_news → articles_clean.parquet 생성
B-2(하이안): assign_month 이후 → model_input.parquet 생성
이용 조건: 빅카인즈 기사 원문은 이용약관상 재배포 제한 → 저장소·제출물에 올리지 않는다.
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""
import os

import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
PROC = f"{cfg['data_dir']}/processed"

"""[H8] 뉴스 감성 (+, 대립 −): 뉴스 수집·정제 부분

담당: 정아영(B1). 점수 산출(score_sentiment, news_sentiment)은 하이안(B2)이 이어서 작성한다.

이론: 기업 관련 기사에 담긴 정보는 이후 실적을 예측하지만 주가가 즉시 다 반영하지 못한다
      (Tetlock, Saar-Tsechansky & Macskassy, 2008). 대립 가설은 비관적 논조 뒤 되돌림(Tetlock, 2007).
자료: D5 빅카인즈 경제지 5곳(머니투데이, 매일경제, 파이낸셜뉴스, 서울경제, 한국경제), 연 단위 엑셀.
이용 조건: 기사 원문은 이용약관상 재배포 제한 → 저장소와 제출물에 올리지 않는다(data/raw는 .gitignore).
           수업·학술 목적으로만 쓰고 모형은 추가 학습하지 않는다(두 담당 교수님 회신, 2026-10-18).
"""


def load_news(raw_dir, codes):
    """[H8] 뉴스 원자료 적재 (+, 대립 −)

    이론: 위 모듈 설명 참조.
    입력: raw_dir — data/raw/news/ 아래 빅카인즈 엑셀(연 단위 파일).
          codes — 대상 종목코드 목록(6자리 문자열, 앞자리 0 유지).
    출력: DataFrame(date, outlet, title, body, excluded_flag, source_file)
    규칙: 일자는 YYYYMMDD(시각 없음). 본문은 200자 이내만 제공된다.
    시점: t월 말까지 공개된 기사만 이후 단계에서 쓴다(여기서는 적재만).
    """
    import glob
    import os
    import warnings

    cols = {"일자": "date", "언론사": "outlet", "제목": "title",
            "본문": "body", "분석제외 여부": "excluded_flag"}
    parts = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # openpyxl 스타일 경고(빅카인즈 파일 형식) 무시
        for f in sorted(glob.glob(os.path.join(raw_dir, "*.xlsx"))):
            d = pd.read_excel(f, usecols=lambda c: c in cols)
            d = d.rename(columns=cols)
            for c in cols.values():
                if c not in d.columns:
                    d[c] = pd.NA
            d["source_file"] = os.path.basename(f)
            parts.append(d)
    if not parts:
        raise FileNotFoundError(f"{raw_dir}에 xlsx 파일이 없다")
    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    return df[["date", "outlet", "title", "body", "excluded_flag", "source_file"]]


def load_name_dict(path="data/processed/company_list.csv"):
    """company_list.csv → {종목코드: [(회사명, first_month, last_month), ...]}
    열 이름은 code/name/first_month/last_month 를 기본으로 하고, 비슷한 이름도 받는다.
    first_month, last_month 는 'YYYY-MM' 문자열, 비어 있으면 제한 없음."""
    cl = pd.read_csv(path, dtype=str, encoding="utf-8")
    pick = lambda *c: next(x for x in c if x in cl.columns)
    ccode = pick("code", "stock_code", "종목코드")
    cname = pick("name", "company_name", "회사명")
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
    """[H8] 분석제외·중복 제거와 제목 기준 종목 매칭 (+, 대립 −)

    입력: df — load_news 출력.
          name_dict — {종목코드: [(회사명, first_month, last_month), ...]}  (load_name_dict 출력).
                      예전 형식 {종목코드: [회사명, ...]} 도 받는다(기간 제한 없음).
    출력: DataFrame(code, date, title, body)
    규칙: ① 분석제외 표시('중복', '예외')가 있는 기사 제거
          ② (제목, 언론사, 일자)가 같은 기사 제거
          ③ 제목에 회사명이 있는 기사만 남김
             - 회사명 앞뒤에 영문·숫자가 바로 붙으면 다른 단어로 본다(SK → SKC 제외)
             - 더 긴 다른 회사명 안에 들어 있는 이름은 그 긴 이름을 지운 뒤 찾는다(SK → SK가스 제외)
             - 기사 날짜의 달이 회사명의 first_month~last_month 안일 때만 그 종목에 붙인다
             - 같은 이름이 같은 달에 두 종목 이상에 해당하면 모호 기사로 보고 뺀다
          ④ 기관 열에만 회사명이 있는 기사는 보류(사용하지 않음)
    기사 수는 신뢰도 점검용으로만 기록하고 예측 변수로 쓰지 않는다.
    """
    import re

    d = df.copy()
    d["title"] = d["title"].fillna("").astype(str).str.replace("車", "차")
    d["body"] = d["body"].fillna("").astype(str).str.replace("車", "차")
    # ① 분석제외 표시('중복', '예외')가 있는 기사 제거
    flag = d["excluded_flag"].fillna("").astype(str)
    d = d[~flag.str.contains("중복|예외")]
    # ② (제목, 언론사, 일자)가 같은 기사 제거
    d = d.dropna(subset=["date"]).drop_duplicates(["title", "outlet", "date"])
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
    return pd.concat(out, ignore_index=True)[cols]


def assign_month(df, trading_days):
    """[H8] 월 귀속 (+, 대립 −)

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


# ---- 아래는 B2(하이안) 담당: classify_type 이후 ----


def classify_type(df):
    """[H8] 기사 유형 분류 (B-1 정아영, 키워드·규칙 확정 후 구현)

    출력: df + news_type 열.
    규칙: 앞선 유형이 우선한다 — 광고 → 공시형 → 논란형 → 화제형 → 기타.
          분석에는 화제형만 쓴다(aggregate_monthly).
    """
    pass


def finbert_probs(texts: pd.Series) -> pd.DataFrame:
    """[H8] KR-FinBert-SC 라벨별 확률 (+, 대립 −)

    입력: texts — 기사 텍스트(제목 + 본문).
    출력: DataFrame(negative, neutral, positive), texts와 같은 인덱스
    규칙: 사전학습 모델을 그대로 쓰고 재학습하지 않는다(이용 조건).
          config.yaml에 text_model_rev(커밋 해시)를 적으면 그 버전으로 고정된다.
    """
    from transformers import pipeline
    clf = pipeline("text-classification", model=cfg["text_model"],
                   revision=cfg.get("text_model_rev"), top_k=None,
                   truncation=True)  # 토큰 제한 512 안전장치
    out = clf(texts.tolist(), batch_size=32)  # 32개씩 묶어 계산
    probs = pd.DataFrame([{d["label"]: d["score"] for d in o} for o in out],
                         index=texts.index)
    return probs[["negative", "neutral", "positive"]]


def score_articles(articles: pd.DataFrame, backend: str = "finbert") -> pd.DataFrame:
    """[H8] 기사별 감성 점수 (+, 대립 −)

    입력: articles — B-1 가공 빅카인즈(code, month, date, title, body[, news_type]).
          backend — "finbert"(config의 text_model). "llm"(upstage ai)은 나중에 추가.
    출력: DataFrame(code, month, date, s[, news_type]), s = P(긍정) − P(부정) ∈ [−1, 1]
    규칙: 제목 + 본문(200자)을 넣는다. 모델 버전 고정, 재학습 없음.
          결과는 article_scores.parquet에 캐시해 재사용한다(원문 열은 저장하지 않음).
          입력 CSV나 backend를 바꾸면 캐시 파일을 지우고 다시 돌린다.
    """
    if backend != "finbert":
        raise NotImplementedError("LLM(Upstage) 백엔드는 나중에 추가")
    cache = f"{PROC}/article_scores.parquet"
    if os.path.exists(cache):  # 캐시가 있으면 재계산하지 않고 바로 읽음
        return pd.read_parquet(cache)

    text = articles["title"].fillna("") + " " + articles["body"].fillna("")
    uniq = text.drop_duplicates()
    print(f"[score_articles] 기사 {len(text)}건 중 고유 텍스트 {len(uniq)}건 감성점수 계산")
    p = finbert_probs(uniq)
    s = (p["positive"] - p["negative"]).set_axis(uniq)

    out = articles.drop(columns=["title", "body"]).assign(s=text.map(s))
    out.to_parquet(cache, index=False)
    return out


def aggregate_monthly(scored: pd.DataFrame) -> pd.DataFrame:
    """[H8] 종목·월 감성 지수 (+, 대립 −)

    입력: scored — score_articles 출력(news_type 열 포함).
    출력: DataFrame(month, code, topic_tone, n_topic)
    규칙: 화제형(news_type == "화제형") 기사만 쓴다.
          topic_tone = 그 달 화제형 기사 s의 평균, n_topic = 화제형 기사 수(점검용, 모델에 넣지 않음).
          기사가 없는 (month, code)는 행이 없으므로 결합 후 결측이 된다.
    """
    topic = scored[scored["news_type"] == "화제형"]
    return (topic.groupby(["month", "code"], as_index=False)  # 같은 달, 종목 기사끼리 묶음
                 .agg(topic_tone=("s", "mean"), n_topic=("s", "size")))


def merge_text_features(panel: pd.DataFrame, text: pd.DataFrame) -> pd.DataFrame:
    """[H8] 정형 패널에 텍스트 변수 결합 (+, 대립 −)

    입력: panel — features.parquet(month, code, 정형 변수, target).
          text — aggregate_monthly 출력.
    출력: panel + (topic_tone, n_topic, no_news)
    규칙: left join (키: month, code / validate="one_to_one"), 결합 전후 행 수가 같은지 확인.
          no_news = 그 달 화제형 기사가 없으면 1 (수집 대상 밖 종목 포함). 이때 n_topic = 0,
          topic_tone은 결측으로 둔다.
    """
    out = panel.merge(text, how="left", on=["month", "code"], validate="one_to_one")
    assert len(out) == len(panel), "결합 후 행 수가 바뀜"
    out["no_news"] = out["n_topic"].isna().astype(int)      # 기사 없음
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


if __name__ == "__main__":
    pass  # 구현 후: 위 함수를 순서대로 호출해 model_input.parquet 저장
