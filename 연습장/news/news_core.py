"""뉴스 처리 핵심 함수 (Colab 노트북과 여기 테스트가 같은 코드를 쓴다)

흐름: 빅카인즈 xlsx 읽기 → 분석제외·중복 제거 → 제목으로 회사 매칭 → 유형 분류(키워드 v1)
      → (Colab) KR-FinBert-SC 점수 → 월 배정(마지막 거래일 규칙) → 종목×월 집계
원문(제목·본문)은 드라이브 안에서만 쓰고, 밖으로 내보내는 파일에는 넣지 않는다.
"""
import re, os, glob, warnings
import pandas as pd, numpy as np

# ---------------- 1. 회사 이름 사전 ----------------
def load_names(search_xlsx):
    """검색목록 '회사·옛이름' 시트 → DataFrame(code, name). 현재 이름 + 옛 이름(' OR ' 구분)."""
    x = pd.read_excel(search_xlsx, sheet_name="회사·옛이름", dtype=str)
    rows = []
    for _, r in x.iterrows():
        code = str(r["종목코드"]).strip().zfill(6)
        names = [r["회사명(현재)"]]
        if isinstance(r["옛 이름 (추정, 확인 필요)"], str):
            names += [s.strip() for s in r["옛 이름 (추정, 확인 필요)"].split(" OR ")]
        for n in names:
            if isinstance(n, str) and len(n.strip()) >= 2:
                rows.append((code, n.strip().replace("車", "차")))
    d = pd.DataFrame(rows, columns=["code", "name"]).drop_duplicates()
    amb = d.groupby("name").code.nunique()
    d = d[d.name.isin(amb[amb == 1].index)]            # 두 종목에 같은 이름 → 모호하므로 제외
    return d

def build_matcher(names):
    """긴 이름부터 찾고, 찾은 자리는 지워서 짧은 이름이 그 안에서 다시 잡히지 않게 한다(SK → SK하이닉스 제외).
    영문·숫자·& 이름은 앞뒤에 영문·숫자·&가 붙으면 다른 단어로 본다(KT → KT&G, KTB 제외)."""
    items = sorted(zip(names.name, names.code), key=lambda t: -len(t[0]))
    alts, code_of = [], {}
    for n, c in items:
        esc = re.escape(n)
        if re.search(r"[A-Za-z0-9]", n):
            esc = r"(?<![A-Za-z0-9&])" + esc + r"(?![A-Za-z0-9&])"
        alts.append(esc); code_of[n] = c
    big = re.compile("|".join(alts))          # 한 번에 훑기: 같은 자리에서는 긴 이름이 먼저 잡힘
    def match(title):
        return sorted({code_of[x.group(0)] for x in big.finditer(title)})
    return match

# ---------------- 2. 기사 읽기 ----------------
COLS = {"뉴스 식별자": "nid", "일자": "date", "언론사": "outlet", "제목": "title", "본문": "body", "분석제외 여부": "excl"}
OUTLETS = {"매일경제", "머니투데이", "서울경제", "파이낸셜뉴스", "한국경제"}

def read_one(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            d = pd.read_excel(path, usecols=lambda c: c in COLS, dtype=str, engine="calamine")
        except Exception:
            d = pd.read_excel(path, usecols=lambda c: c in COLS, dtype=str)
    d = d.rename(columns=COLS)
    for c in COLS.values():
        if c not in d: d[c] = None
    d["source_file"] = os.path.relpath(path).split(os.sep)[-2] + "/" + os.path.basename(path)
    return d[list(COLS.values()) + ["source_file"]]

def clean(d):
    n0 = len(d)
    d = d[~d.excl.fillna("").str.contains("중복|예외")]                       # 빅카인즈 분석제외 표시
    d = d[d.outlet.isin(OUTLETS)]                                              # 언론사 5곳만
    d["date"] = pd.to_datetime(d.date.astype(str).str[:8], format="%Y%m%d", errors="coerce")
    d = d.dropna(subset=["date"])
    d["title"] = d.title.fillna("").str.replace("車", "차"); d["body"] = d.body.fillna("")
    d = d.drop_duplicates("nid").drop_duplicates(["title", "outlet", "date"])  # OR 묶음 검색끼리 겹친 기사 제거
    return d, n0

# ---------------- 3. 유형 분류 (키워드 v1) ----------------
# 우선순위: 광고 → 공시형 → 논란형 → 화제형 → 기타.  제목 기준(본문은 200자라 잡음이 많음).
KW = {
    "광고": ["[광고]", "[AD]", "[애드]", "협찬", "기획광고", "[PR]"],
    "공시형": ["유상증자", "무상증자", "증자", "감자", "실적", "영업이익", "영업익", "매출", "순이익", "순익", "흑자", "적자",
              "어닝", "배당", "수주", "공급계약", "단일판매", "특징주", "공시", "목표주가", "목표가", "투자의견", "자사주",
              "신용등급", "상한가", "하한가", "신고가", "신저가", "52주", "시총", "시가총액", "주가", "급등", "급락",
              "순매수", "순매도", "코스피", "증시", "IR", "결산", "분기", "주총", "주주총회", "액면분할", "상장"],
    "논란형": ["횡령", "배임", "소송", "리콜", "압수수색", "파업", "과징금", "검찰", "기소", "구속", "갑질", "논란", "의혹",
              "제재", "적발", "담합", "유출", "불매", "고발", "징계", "비자금", "분식", "탈세", "사고", "폭발", "화재",
              "결함", "피소", "벌금", "공정위", "부당", "불공정", "노조"],
    "화제형": ["신사업", "신제품", "출시", "제휴", "협약", "MOU", "업무협약", "투자", "개발", "진출", "공개", "선보", "론칭",
              "런칭", "인수", "합병", "M&A", "특허", "수출", "기술", "공장", "착공", "준공", "신규", "확대", "상생", "협력",
              "파트너", "선정", "수상", "1위", "세계 최초", "국내 최초", "첫", "도입", "가동", "개소", "오픈", "합작", "개관", "획득", "인증", "확보"],
}
TYPES = ["광고", "공시형", "논란형", "화제형"]

def classify(title):
    for t in TYPES:
        if any(k in title for k in KW[t]):
            return t
    return "기타"

# ---------------- 4. 월 배정 ----------------
def assign_month(dates):
    """마지막 거래일(평일 기준 근사, 한국 공휴일 미반영) 당일과 그 이후 기사는 다음 달로.
    이유: t월 수익률은 마지막 거래일 종가까지라, 그날 기사는 t월 말 정보로 보기 어렵다."""
    per = dates.dt.to_period("M")
    last_bday = (dates + pd.offsets.BMonthEnd(0)).dt.normalize()
    return per.where(dates < last_bday, per + 1).astype(str)

# ---------------- 5. 종목×월 집계 ----------------
def aggregate(scored):
    """scored: code, month, type, s(=P긍정−P부정). 출력: 종목×월 요약(원문 없음)."""
    g = scored.groupby(["month", "code"])
    out = g.size().rename("n_all").to_frame()
    for t, k in [("화제형", "topic"), ("공시형", "disc"), ("논란형", "contro"), ("기타", "etc")]:
        sub = scored[scored.type == t].groupby(["month", "code"])
        out[f"n_{k}"] = sub.size()
        out[f"{k}_tone"] = sub.s.mean()
    out["all_tone"] = scored.dropna(subset=["s"]).groupby(["month", "code"]).s.mean()
    sub = scored[scored.type == "화제형"].groupby(["month", "code"])
    out["topic_pos_share"] = sub.s.apply(lambda x: (x > 0).mean())
    for c in [c for c in out if c.startswith("n_")]:
        out[c] = out[c].fillna(0).astype(int)
    return out.reset_index()
