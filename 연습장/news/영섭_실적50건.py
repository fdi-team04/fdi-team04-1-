"""실적 테마 사람 확인용 제목 50건 뽑기 (VS Code, 1분)
왜: 테마 판정에서 '실적 호재'가 통과 → 키워드가 진짜 실적 호재 기사를 잡는지 사람이 확인 (정밀도 80% 넘는지)
같은 폴더: 이 파일, theme_rules.py, articles_clean.csv
실행: python 영섭_실적50건.py articles_clean.csv
결과: 실적테마_사람확인_50건.xlsx  → 아영에게 전달 (기사 제목이 들어 있으니 깃·공개 채널엔 올리지 말 것)
뽑는 법: 실적_호재로 잡힌 기사에서 기간별(2001–10 / 2011–15 / 2016–21) 비율대로, 같은 제목은 하나만,
        한 회사는 최대 3건, 무작위(시드 고정 1120 → 다시 돌려도 같은 50건)
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme_rules import themes_of
SRC = next((a for a in sys.argv[1:] if not a.startswith("--")), "articles_clean.csv")
OUT = os.path.join(os.path.dirname(os.path.abspath(SRC)), "실적테마_사람확인_50건.xlsx"); N = 50; SEED = 1120
head = pd.read_csv(SRC, nrows=0).columns
pick = lambda *c: next((x for x in c if x in head), None)
C = {"code": pick("code", "종목코드"), "date": pick("date", "일자"), "title": pick("title", "제목")}
nm = pick("name", "company", "회사명", "종목명")
use = list(C.values()) + ([nm] if nm else [])
x = pd.read_csv(SRC, usecols=use, dtype=str).rename(columns={v: k for k, v in C.items()})
if nm: x = x.rename(columns={nm: "name"})
x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
x = x[x.date.notna() & (x.date.dt.year <= 2021)]
x = x[x.title.map(lambda t: "실적_호재" in themes_of(t))].drop_duplicates("title")
print(f"실적_호재로 잡힌 기사 {len(x):,}건")
x["기간"] = pd.cut(x.date.dt.year, [2000, 2010, 2015, 2021], labels=["2001–10", "2011–15", "2016–21"])
x = x.sample(frac=1, random_state=SEED)
x = x[x.groupby("code").cumcount() < 3]                                   # 한 회사 최대 3건
share = x.기간.value_counts(normalize=True)
k = (share * N).round().astype(int); k.iloc[0] += N - k.sum()             # 합이 정확히 50
s = pd.concat([x[x.기간 == p].head(n) for p, n in k.items()]).sample(frac=1, random_state=SEED).reset_index(drop=True)
out = pd.DataFrame({"번호": range(1, len(s) + 1), "일자": s.date.dt.strftime("%Y-%m-%d"), "기간": s.기간.astype(str),
                    "종목코드": s.code, **({"회사": s["name"]} if "name" in s else {}), "제목": s.title,
                    "규칙 판정": "실적_호재", "사람 판정(맞음/아님/애매)": "", "아니면 이유": ""})
guide = pd.DataFrame({"판단 기준": [
    "맞음: 그 회사의 실적이 좋았다는 소식 (발표된 실적, 흑자전환, 사상 최대, 어닝 서프라이즈 등 이미 나온 결과)",
    "아님: 다른 회사 얘기 / 실적이 아니라 주가·업황·정책 얘기 / 여러 회사를 나열한 시황 기사에서 이 회사가 주인공이 아님",
    "아님: 실적이 좋아질 거라는 전망·기대만 있는 기사 (아직 결과가 아님)",
    "애매: 위로 판단이 안 되면 '애매'로 두고 이유를 적기",
    "목표: 50건 중 '맞음' 비율이 80% 이상이면 키워드 규칙 통과",
    "제목만 보고 판단 (규칙도 제목만 봄). 원문 검색하지 않아도 됨"]})
try:
    with pd.ExcelWriter(OUT) as w:
        out.to_excel(w, sheet_name="확인할 50건", index=False); guide.to_excel(w, sheet_name="판단 기준", index=False)
    print(f"끝! {OUT} ({len(out)}건) → 아영에게 전달")
except ModuleNotFoundError:
    p = OUT.replace(".xlsx", ".csv"); out.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"엑셀 모듈(openpyxl)이 없어 CSV로 저장: {p}  (엑셀로 원하면: pip install openpyxl 후 다시 실행)")
print(out.기간.value_counts().to_dict())
