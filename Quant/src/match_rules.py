"""제목 회사 매칭 규칙 v2 (파일럿 27만 건 눈 점검으로 만든 규칙) — 아영 정리 코드에 그대로 붙일 수 있게 함수 하나로

왜 필요한가 (파일럿에서 실제로 나온 오류):
 1. 짧은 이름 뒤에 다른 말이 붙음  : 'SK텔'(SK텔레콤), 'CJ오쇼핑', 'KCC건설', '두산重', '대우조선', 'CJ證' → 모회사가 아님
 2. 이름 앞에 글자가 붙음          : '광고대상'→대상, '전효성'→효성, 'SKC코오롱PI'→코오롱, '최신원'→신원
 3. 증권사 리포트 출처 꼬리표       : '...목표가↑-대우', '-동양證', '-SK' → 리포트를 쓴 증권사이지 기사 주인공이 아님
 4. 이름을 쓰던 기간 밖              : '대우'(2001 상장폐지) 이후 기사, '신한은행'(2001 지주 편입) 이후 → 다른 종목
규칙:
 - 이름 앞 글자가 한글·영문·숫자면 무시
 - 이름 바로 뒤 글자가 조사·문장부호·숫자·허용 꼬리말(사장, 회장, 그룹, 株 등)이 아니면 무시
 - 제목 끝 '-증권사' 꼬리표는 매칭 전에 지움
 - company_list의 name_first~name_last(없으면 그 종목 상장기간) 밖이면 무시
남는 한계: '두산에 9대8'(야구단), '태평양항로'처럼 일반 단어와 같은 이름은 일부 남음 → 사람 점검 표본으로 비율만 보고
"""
import re

OK_NEXT_CHARS = set("은는이가을를의와과도에엔로만,.·ㆍ'\"‘’“”…?!()[]<>-~/:; ") | set("株社側號측등㈜白장")   # 장: '외환은행장'처럼 은행장·사장
OK_NEXT_WORDS = ("사장", "회장", "부회장", "대표", "그룹", "노조", "주가", "장중", "보다", "까지", "부터", "처럼", "만큼", "마저", "조차", "에서", "으로")
REPORT_TAIL = re.compile(r"[-–]\s*[^\s-]*(證|증권|투자|대우|현대|대신|동부|하이|흥국|KTB|KB|유진|신영|미래|한투|NH|SK|IBK|키움|메리츠|이베스트|하나|신한|교보|삼성|한화|동양|CJ|GS|골드만삭스|굿모닝|한누리|CSFB|맥쿼리|씨티|UBS|모건|골드만|JP모간|메릴|노무라|다이와|CLSA|HMC|부국|한양|SK증권)\s*$")


def norm_title(t):
    return REPORT_TAIL.sub("", str(t).replace("＆", "&").replace("車", "차"))


GENERIC = {"대상", "태평양", "동양", "대우", "동원", "미원", "고제", "신원", "디아이", "서흥", "백산", "화신", "삼립", "동서", "진로", "대교"}   # 일반 단어와 같은 이름


def name_hit(title, name):
    """제목(norm_title 거친 것)에 name이 '독립된 회사 이름'으로 나오면 True"""
    for m in re.finditer(re.escape(name), title):
        prev, rest = title[m.start() - 1:m.start()], title[m.end():]
        if prev and re.match(r"[가-힣A-Za-z0-9&]", prev):
            continue
        if name in GENERIC:   # '디자인 대상', '태평양 연안' 같은 일반 단어를 거르려고: 제목 맨 앞/[ ] 뒤이거나, 바로 뒤가 쉼표·따옴표·괄호·숫자일 때만
            if not (m.start() == 0 or prev in "]】)" or re.match(r"[,(‘'\"·]|\s*\d", rest)):
                continue
        if not rest or rest[0] in OK_NEXT_CHARS or rest[0].isdigit() or rest.startswith(OK_NEXT_WORDS):
            return True
    return False


def build_index(company_list):
    """company_list(DataFrame: code,name,first_month,last_month,name_first,name_last) → {name: [(code, from, to)]}"""
    c = company_list.fillna("").copy()
    c["lf"] = c.groupby("code").first_month.transform("max")
    c["ll"] = c.groupby("code").last_month.transform("max")
    idx = {}
    for r in c.itertuples():
        f = r.name_first or r.lf or "0000-00"; l = r.name_last or r.ll or "9999-12"
        idx.setdefault(r.name, []).append((r.code, f, l))
    return idx


def match_title(title, ym, idx, names_sorted=None):
    """제목 하나 → 붙일 종목코드 집합. ym = 'YYYY-MM' (기사 월). 긴 이름 먼저 보고, 쓰인 자리는 지워서 짧은 이름이 다시 잡지 않게 함"""
    t = norm_title(title); out = set()
    for n in names_sorted or sorted(idx, key=len, reverse=True):
        if n not in t or not name_hit(t, n):
            continue
        owners = [c for c, f, l in idx[n] if f <= ym <= l]
        if len(owners) == 1:
            out.add(owners[0])
        t = t.replace(n, "□" * len(n))   # 'SK하이닉스'를 잡은 뒤 그 안의 'SK'를 또 잡지 않게
    return out
