"""[03] 정리된 기사 → 목표가 상향 종목·월 표   담당: B-2(하이안)

입력  data/processed/articles_clean.csv   정리·종목 매칭이 끝난 기사 (아영 정리 코드 출력, 원문 포함 → 비공개)
                                           2022–2025분이 생기면 config.yaml의 articles 목록에 추가한다
      data/raw/Structured/*__raw_month_end_market_cap.csv   배포 시총 (수집 대상을 정할 때만 쓴다)
      tables/last_trading_days.csv        월별 마지막 거래일 (src/make_trading_days.py로 한 번 만든다)
출력  tables/tp_monthly.csv               종목·월 표 (원문 없음 → 저장소에 올린다). 예원님 합치기 코드가 정형 데이터에 붙인다

흐름: 기사 적재 → 월 배정 → 목표가 상향 라벨(label_tp, 키워드 규칙) → 수집 대상(전년 12월 말 시총 상위 200) × 그해 12개월 표에 집계
기사 원문이 없는 환경(팀원 Docker·채점)에서는 표를 새로 만들지 않고 저장소의 tables/tp_monthly.csv를 쓴다.
이용 조건: 빅카인즈 기사 원문은 재배포 제한 → 저장소·제출물에 올리지 않는다(두 담당 교수님 회신, 2026-10-18).
"""
import glob
import os
import re

import pandas as pd
import yaml

cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
DATA = cfg["data_dir"]
OUT = f"{cfg['tables_dir']}/tp_monthly.csv"
MCAP = f"{DATA}/raw/Structured/*__raw_month_end_market_cap.csv"  # warmup·train·test 배포 시총
TOP_N = 200  # 수집 대상: 전년 12월 말 시총 상위 200 (11/22 결정, 그해가 시작되기 전에 알 수 있는 정보)
# 목표가 상향 키워드 v0: 연습장 news/theme_rules.py의 '목표가_상향'과 같다 (v1에서도 이 테마는 그대로, 제외 패턴 없음)
TP_WORDS = ["목표가↑", "목표가 상향", "목표주가 상향", "목표주가↑", "투자의견 상향", "매수로 상향"]


def label_tp(titles: pd.Series) -> pd.Series:
    """제목 → 목표가 상향 기사면 True (연습장 theme_rules.themes_of와 같은 규칙: TP_WORDS 중 하나가 제목에 그대로 있으면)

    사람 확인 50건(아영·예원 합의) 정밀도 100%. 임베딩 등 다른 분류로 바꿀 때는 이 함수만 바꾼다.
    """
    return titles.fillna("").astype(str).str.contains("|".join(map(re.escape, TP_WORDS)))


def load_articles(paths: list[str]) -> pd.DataFrame:
    """정리된 기사 → DataFrame(code, date, title). 열 이름은 영문(code·date·title)과 빅카인즈 한글 둘 다 받는다."""
    names = {"종목코드": "code", "일자": "date", "제목": "title"}
    keep = lambda c: c in names or c in names.values()  # noqa: E731  본문 등 나머지 열은 읽지 않는다
    a = pd.concat([pd.read_csv(p, usecols=keep, dtype=str) for p in paths], ignore_index=True).rename(columns=names)
    a["code"] = a["code"].str.replace(r"\.0$", "", regex=True).str.zfill(6)  # 종목코드는 문자열 6자리
    a["date"] = pd.to_datetime(a["date"].str[:10].str.replace("-", ""), format="%Y%m%d", errors="coerce")
    return a.dropna(subset=["date"])

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
        try:
            import python_calamine  # noqa: F401
            engine = "calamine"
        except ImportError:
            engine = None
        files = sorted(glob.glob(os.path.join(raw_dir, "*.xlsx")))
        for k, f in enumerate(files, 1):
            if k % 25 == 0 or k == len(files):
                print(f"[load_news] {k}/{len(files)} 파일 읽는 중", flush=True)
            d = pd.read_excel(f, usecols=lambda c: c in cols, engine=engine)
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


def dedupe_news(df):
    t_all = df["title"].fillna("").astype(str).str.replace("車", "차")
    flag = df["excluded_flag"].fillna("").astype(str)
    ok = (~flag.str.contains("중복|예외")) & df["date"].notna()
    del flag
    key = pd.DataFrame({"title": t_all[ok], "outlet": df["outlet"][ok], "date": df["date"][ok]})
    keep_idx = key.drop_duplicates().index
    del key
    d = df.loc[keep_idx].copy()
    d["title"] = t_all.loc[keep_idx]
    del t_all
    d["body"] = d["body"].fillna("").astype(str).str.replace("車", "차")
    d = d.reset_index(drop=True)
    print(f"[dedupe_news] 제거·중복 정리 후 기사: {len(d)}건", flush=True)
    return d


def assign_month(dates: pd.Series, last_days: pd.Series) -> pd.Series:
    """기사 날짜 → 귀속 월 "YYYY-MM"

    t월 마지막 거래일 전날까지의 기사는 t월, 마지막 거래일 당일과 그 뒤 기사는 t+1월로 보낸다.
    기사에 시각이 없어 당일 기사가 장 마감 뒤일 수 있으므로 보수적으로 다음 달로 보낸다(미래 정보 차단).
    """
    last = pd.to_datetime(pd.Series(list(last_days)))
    per = dates.dt.to_period("M")
    cut = per.map(last.groupby(last.dt.to_period("M")).max())  # 그 달 마지막 거래일
    if cut.isna().any():
        raise ValueError(f"last_trading_days에 없는 달의 기사가 있다: {sorted(per[cut.isna()].astype(str).unique())}")
    return per.where(dates < cut, per + 1).astype(str)


def news_frame() -> pd.DataFrame:
    """수집 대상 칸 (month, code): 전년 12월 말 시총 상위 TOP_N 종목 × 그해 1~12월

    그달 기사가 0건이어도 행이 있다 → "수집했지만 기사 없음"(tp_up = 0)과 "수집 안 함"(표에 없음)이 구분된다.
    """
    mc = pd.concat([pd.read_csv(f, index_col="month") for f in sorted(glob.glob(MCAP))])
    dec = mc[mc.index.str.endswith("-12")]
    parts = [pd.MultiIndex.from_product(
        [pd.period_range(f"{int(m[:4]) + 1}-01", periods=12, freq="M").astype(str), row.nlargest(TOP_N).index],
        names=["month", "code"]).to_frame(index=False) for m, row in dec.iterrows()]
    return pd.concat(parts, ignore_index=True)


def build_tp_table(articles: pd.DataFrame, frame: pd.DataFrame, last_days: pd.Series) -> pd.DataFrame:
    """종목·월 표: month, code, news_missing(=0), tp_up, n_tp_up, n_news

    tp_up = 그달 목표가 상향 기사가 1건이라도 있으면 1 (본 변수). n_tp_up·n_news는 참고(점검)용.
    기사를 받은 마지막 날이 속한 달 뒤는 "수집 안 함"이므로 표에서 뺀다(빈 달을 기사 0건으로 읽지 않게).
    """
    a = articles.assign(month=assign_month(articles["date"], last_days), tp=label_tp(articles["title"]))
    cnt = a.groupby(["month", "code"]).agg(n_news=("tp", "size"), n_tp_up=("tp", "sum")).reset_index()
    end = str(articles["date"].max().to_period("M"))
    t = frame[frame["month"] <= end].merge(cnt, how="left", on=["month", "code"])
    t[["n_news", "n_tp_up"]] = t[["n_news", "n_tp_up"]].fillna(0).astype(int)
    t = t.assign(news_missing=0, tp_up=(t["n_tp_up"] > 0).astype(int))
    print(f"[03] 기사 {len(a):,}건 중 수집 대상 칸에 들어간 기사 {t['n_news'].sum():,}건 | "
          f"표 {len(t):,}칸 ({t['month'].min()} ~ {t['month'].max()}), tp_up = 1 비율 {t['tp_up'].mean():.1%}")
    return t[["month", "code", "news_missing", "tp_up", "n_tp_up", "n_news"]]


if __name__ == "__main__":
    paths = [f"{DATA}/{p}" for p in cfg["articles"]]
    days = f"{cfg['tables_dir']}/last_trading_days.csv"
    missing = [p for p in paths + [days] if not os.path.exists(p)]
    if missing:  # 팀원 Docker·채점 환경: 기사 원문이 없다
        print(f"[03] 없는 파일 {missing} → 표를 새로 만들지 않고 저장소의 {OUT}를 그대로 쓴다")
    else:
        build_tp_table(load_articles(paths), news_frame(), pd.read_csv(days)["date"]).to_csv(OUT, index=False)
        print(f"[03] {OUT} 저장")
