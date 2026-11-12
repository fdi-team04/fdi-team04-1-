"""상위 200 뉴스 검색 목록 만들기 (매크로용)
대상: 매년 직전 연도 12월 말 시총 상위 200 (검색 연도 Y = 2001..2021, 기준 = Y-1년 12월)
이미 받은 범위: 상위100 검색목록의 (종목, 기간)
출력: 연습장/news/뉴스수집_검색목록_상위200.xlsx
"""
import pandas as pd, re
OUT = "/home/user/fdi-team04-1-/연습장"
f = pd.read_parquet(f"{OUT}/out/features.parquet")
d = f[f.month.str[5:] == "12"].copy(); d["rk"] = d.groupby("month").me.rank(ascending=False)
d = d[(d.month >= "2000-12") & (d.month <= "2020-12") & (d.rk <= 200)]
d["year"] = d.month.str[:4].astype(int) + 1
j = pd.read_csv("/home/user/data/joined_investor.csv", dtype={"Stock_Code": str}, usecols=["Stock_Code", "company_name"]).dropna()
nm = j.drop_duplicates("Stock_Code").set_index("Stock_Code").company_name.to_dict()

S = "/home/user/data/search_list.xlsx"
s = pd.read_excel(S, sheet_name="회사·옛이름", dtype=str); s["code"] = s["종목코드"].str.zfill(6)
old_names = {r.code: (r["회사명(현재)"], r["옛 이름 (추정, 확인 필요)"] if isinstance(r["옛 이름 (추정, 확인 필요)"], str) else "") for _, r in s.iterrows()}

# 새 회사 옛 이름 (추정, 확인 필요)
NEW_OLD = {
 "000070": "삼양사", "000370": "신동아화재", "000400": "대한화재", "000480": "조선내화", "000540": "쌍용화재",
 "000680": "국제상사", "000990": "동부하이텍 OR 동부일렉트로닉스 OR 동부아남반도체", "001060": "중외제약",
 "001250": "쌍용", "001500": "HMC투자증권 OR 신흥증권", "001520": "동양메이저", "001630": "종근당",
 "001680": "미원", "001880": "삼호 OR 대림건설", "001940": "한국철강", "002030": "아세아시멘트",
 "002270": "롯데삼강", "002620": "제일약품", "003030": "세아제강 OR 부산파이프", "003070": "코오롱건설",
 "003530": "한화증권", "003570": "S&T중공업 OR 통일중공업", "003640": "연합철강", "003850": "보령제약",
 "004130": "대덕지디에스", "004560": "삼미특수강", "004710": "한솔전자", "005620": "대성산업",
 "005740": "크라운제과", "005850": "삼립산업", "006390": "현대시멘트", "007340": "동아타이어 OR DTR오토모티브",
 "007700": "F&F", "007860": "한일이화", "008000": "웅진케미칼 OR 새한", "009180": "한솔CSN",
 "009200": "신무림제지", "009580": "동해펄프", "009680": "한국베랄", "009970": "영원무역",
 "010780": "동서산업", "011760": "현대종합상사", "013570": "동양기전", "014680": "한솔화학",
 "014790": "한라건설", "016380": "동부제철", "016420": "세종증권", "016450": "한세실업",
 "016610": "동부증권 OR DB금융투자", "016800": "한샘퍼시스", "016880": "웅진홀딩스", "017390": "서울도시가스",
 "017810": "풀무원홀딩스", "017940": "LG칼텍스가스", "018670": "유공가스", "020760": "일진디스플레이",
 "021960": "우리파이낸셜", "024720": "한국콜마", "025000": "한국포리올", "025540": "한국단자공업",
 "033660": "아주캐피탈", "034310": "한국신용정보 OR NICE홀딩스", "035000": "지투알 OR LG애드",
 "036530": "S&T홀딩스", "037560": "CJ헬로 OR CJ헬로비전", "044820": "코스맥스", "053210": "KT스카이라이프",
 "058430": "포스코강판", "064960": "S&T대우 OR 대우정밀", "069260": "휴켐스", "079430": "리바트",
 "082740": "HSD엔진 OR 두산엔진", "084870": "베이직하우스", "085620": "SK생명", "093050": "LG패션",
 "101060": "SBS홀딩스", "108670": "LG하우시스", "170900": "동아제약", "181710": "NHN엔터테인먼트",
 "267260": "현대일렉트릭", "267270": "현대건설기계", "280360": "롯데제과", "336370": "두산솔루스",
 "363280": "태영건설",
}

# 이미 받은 (종목, 연도)
done = set()
for sh in ["1번", "2번", "3번", "4번"]:
    v = pd.read_excel(S, sheet_name=sh, dtype=str)
    for _, r in v.iterrows():
        try:
            y0, y1 = int(str(r["검색 시작일"])[:4]), int(str(r["검색 종료일"])[:4])
        except Exception:
            continue
        for c in re.findall(r"\d{4,6}", str(r["종목코드"])):
            for y in range(y0, y1 + 1): done.add((c.zfill(6), y))

def names(code):
    if code in old_names:
        cur, old = old_names[code]; src = "기존 목록"
    else:
        cur, old = nm.get(code, ""), NEW_OLD.get(code, ""); src = "새로 추가(추정)" if code in NEW_OLD else "새로 추가"
    q = cur + ("" if not old else " OR " + old)
    return cur, old, q, src

info = d.groupby("code").agg(최고순위=("rk", "min"), 해수=("year", "size")).reset_index()
yrs = d.groupby("code").year.apply(sorted).to_dict()

def runs(ys):  # 연속 구간 → 5년 이하로 자르기
    out, a = [], ys[0]; p = a
    for y in ys[1:] + [None]:
        if y is None or y != p + 1:
            s0 = a
            while s0 <= p:
                e = min(s0 + 4, p); out.append((s0, e)); s0 = e + 1
            a = y
        if y is not None: p = y
    return out

rows = []
for code in sorted(yrs):
    need = [y for y in yrs[code] if (code, y) not in done]
    cur, old, q, src = names(code)
    for a, b in runs(yrs[code]):
        rows.append({"code": code, "a": a, "b": b, "need": all((code, y) not in done for y in range(a, b + 1)),
                     "part": any((code, y) not in done for y in range(a, b + 1))})
R = []
for code in sorted(yrs):
    cur, old, q, src = names(code)
    need = [y for y in yrs[code] if (code, y) not in done]
    for a, b in (runs(need) if need else []):
        R.append({"종목코드": code, "회사명(현재)": cur, "검색어 (옛 이름 포함, 복사용)": q,
                  "검색 시작일": f"{a}-01-01", "검색 종료일": f"{b}-12-31",
                  "구분": "새 회사" if code not in old_names else "기존 회사 · 추가 기간",
                  "옛 이름 출처": src, "파일명": f"{code}_{a}-{b}.xlsx", "실제 건수": "", "완료(O/X)": "", "메모": ""})
todo = pd.DataFrame(R); todo.insert(0, "순번", range(1, len(todo) + 1))

C = []
for _, r in info.sort_values("최고순위").iterrows():
    cur, old, q, src = names(r.code); ys = yrs[r.code]
    C.append({"종목코드": r.code, "회사명(현재)": cur, "옛 이름 (추정, 확인 필요)": old, "검색어 (복사용)": q,
              "상위200에 있던 해": f"{min(ys)}~{max(ys)} 중 {len(ys)}년", "최고 순위": int(r.최고순위),
              "상위100 목록에 있음": "O" if r.code in old_names else "",
              "이미 받은 해 수": sum((r.code, y) in done for y in ys), "남은 해 수": sum((r.code, y) not in done for y in ys),
              "옛 이름 출처": src})
comp = pd.DataFrame(C)

guide = pd.DataFrame({"항목": ["대상", "기준", "언론사", "기간", "검색어", "나눠 받기", "파일명", "이미 받은 것", "옛 이름", "금지"],
  "내용": ["매년 직전 연도 말 시가총액 상위 200개 회사 (2001~2021, 478개사)",
          "검색 연도 Y의 대상 = Y−1년 12월 말 시총 순위 (오늘 기준 아님 → 미래 정보 방지)",
          "매일경제, 머니투데이, 서울경제, 파이낸셜뉴스, 한국경제 5곳만",
          "'남은 검색목록' 시트의 시작일~종료일 (한 줄 최대 5년, 그 회사가 상위 200이던 해만)",
          "'검색어 (옛 이름 포함, 복사용)' 칸을 그대로 사용. 분류는 건드리지 않기",
          "한 번에 2만 건 넘으면 기간을 반으로 나눠 받고 파일명 끝에 a, b",
          "종목코드6자리_시작연도-끝연도.xlsx (예: 000370_2001-2005.xlsx)",
          "상위100 검색목록으로 이미 받은 (회사, 연도)는 빼고 만들었음. 그 기간은 다시 받지 않기",
          "기존 242개사는 예전 목록 그대로. 새 회사 옛 이름은 기억에 의존한 추정이라 첫 검색 전 회사 연혁으로 확인",
          "자동 수집 전 빅카인즈 이용 조건 확인. 받은 엑셀은 GitHub·공개방에 올리지 않기"]})
fn = f"{OUT}/news/뉴스수집_검색목록_상위200.xlsx"
with pd.ExcelWriter(fn, engine="openpyxl") as w:
    guide.to_excel(w, sheet_name="안내", index=False); todo.to_excel(w, sheet_name="남은 검색목록", index=False)
    comp.to_excel(w, sheet_name="회사·옛이름 (478)", index=False)
    for ws in w.book.worksheets:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in col) * 1.3))
        ws.freeze_panes = "A2"
print("남은 줄", len(todo), "| 새 회사", (todo.구분 == "새 회사").sum(), "| 기존 추가", (todo.구분 != "새 회사").sum())
print("회사", len(comp), "| 새 회사 중 옛이름 추정 채움", sum(1 for c in comp.itertuples() if c._10 == "새로 추가(추정)"))
print("이미 받은 (회사,연도)", len(done), "| 상위200 (회사,연도) 전체", sum(len(v) for v in yrs.values()), "| 남은", sum(len([y for y in v if (k, y) not in done]) for k, v in yrs.items()))
