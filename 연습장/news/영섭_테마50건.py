"""테마 사람 확인용 제목 50건 뽑기 (아무 테마나, VS Code, 1분)
같은 폴더: 이 파일, theme_rules.py, articles_clean.csv
실행: python 영섭_테마50건.py articles_clean.csv 목표가_상향      (마지막에 theme_rules.py의 테마 이름)
      새 표본: python 영섭_테마50건.py articles_clean.csv 실적_호재 --seed=1121
결과: <테마>_사람확인_50건.xlsx → 아영에게 전달 (기사 제목이 들어 있으니 깃·공개 채널엔 올리지 말 것)
뽑는 법: 그 테마로 잡힌 기사에서 기간별(2001–10 / 2011–15 / 2016–21) 비율대로, 같은 제목은 하나만,
        한 회사는 최대 3건, 무작위(시드 고정 1120 → 다시 돌려도 같은 50건)
"""
import os, sys, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from theme_rules import themes_of
from theme_rules import THEMES
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SRC = ARGS[0] if ARGS else "articles_clean.csv"; TH = ARGS[1] if len(ARGS) > 1 else "목표가_상향"
if TH not in THEMES: sys.exit(f"테마 이름이 없어요: {TH}  (가능: {', '.join(THEMES)})")
SEED = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--seed=")), 1120))   # 새 표본이 필요하면 --seed=1121 처럼
OUT = os.path.join(os.path.dirname(os.path.abspath(SRC)), f"{TH}_사람확인_50건" + ("" if SEED == 1120 else f"_seed{SEED}") + ".xlsx"); N = 50
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
x = x[x.title.map(lambda t: TH in themes_of(t))].drop_duplicates("title")
print(f"{TH}로 잡힌 기사 {len(x):,}건")
x["기간"] = pd.cut(x.date.dt.year, [2000, 2010, 2015, 2021], labels=["2001–10", "2011–15", "2016–21"])
x = x.sample(frac=1, random_state=SEED)
x = x[x.groupby("code").cumcount() < 3]                                   # 한 회사 최대 3건
share = x.기간.value_counts(normalize=True)
k = (share * N).round().astype(int); k.iloc[0] += N - k.sum()             # 합이 정확히 50
s = pd.concat([x[x.기간 == p].head(n) for p, n in k.items()]).sample(frac=1, random_state=SEED).reset_index(drop=True)
out = pd.DataFrame({"번호": range(1, len(s) + 1), "일자": s.date.dt.strftime("%Y-%m-%d"), "기간": s.기간.astype(str),
                    "종목코드": s.code, **({"회사": s["name"]} if "name" in s else {}), "제목": s.title,
                    "규칙 판정": TH, "사람 판정(맞음/아님/애매)": "", "아니면 이유": ""})
CRIT = {
    "실적_호재": ["맞음: 그 회사의 실적이 좋았다는 소식 (발표된 실적, 흑자전환, 사상 최대, 어닝 서프라이즈 등 이미 나온 결과)",
                 "아님: 다른 회사 얘기 / 실적이 아니라 주가·업황·정책 얘기 / 여러 회사를 나열한 시황 기사에서 이 회사가 주인공이 아님",
                 "아님: 실적이 좋아질 거라는 전망·기대만 있는 기사 (아직 결과가 아님)"],
    "목표가_상향": ["맞음: 증권사가 그 회사의 목표주가 또는 투자의견을 올렸다는 기사 (예: 'OO, 목표가↑-XX증권')",
                  "아님: 목표가를 유지·내린 기사 / 다른 회사의 목표가 / 증권사 자체(증권주)에 대한 기사인데 회사가 다름",
                  "아님: 목표가 얘기가 아닌데 '상향' 단어만 들어간 기사 (예: 신용등급, 실적 전망 상향만 있는 경우)"],
}
guide = pd.DataFrame({"판단 기준": CRIT.get(TH, [f"맞음: 제목이 '{TH}' 테마의 뜻에 맞는 그 회사 소식", "아님: 다른 회사 얘기이거나 테마와 관련 없는 기사"]) + [
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
