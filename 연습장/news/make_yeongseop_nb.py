"""영섭_분류·점수 Colab 노트북 생성"""
import json
rules = open("news_rules.py", encoding="utf-8").read()
steps = open("yeongseop_steps.py", encoding="utf-8").read()
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
cells = [
md("""# 영섭: 2-3 분류 시험 + 2-4 감성 점수 (Colab)

**순서** (위에서부터 실행)
1. 설정: 아영님 `articles_clean_100.csv` 경로만 바꾸기
2. 함수 불러오기 (키워드 규칙 v1, 뽑기·일치율·점수 함수)
3. **[1] 100건 뽑기** → 예원님·아영님용 정답지 2개 (컴퓨터 답 없음)
4. **[3] 일치율** → 두 사람 정답을 받은 뒤 실행 (사람끼리 일치율, 컴퓨터 일치율, 화제형 정밀도)
5. **2-4 감성 점수** → 100건으로 먼저 점검, 그다음 전체 → 회사×월 파일

런타임: 4~5번은 **T4 GPU** 권장. 원문(제목·본문)이 들어간 파일은 비공개 드라이브에만 둘 것."""),
code("""from google.colab import drive
drive.mount('/content/drive')
CLEAN = '/content/drive/MyDrive/뉴스_원몬_비공개/articles_clean_100.csv'   # ← 아영님 파일 위치로 바꾸기
OUT   = '/content/drive/MyDrive/4팀_분류시험'                              # 결과 저장 폴더
PREV_TEST = ''   # 지난 시험 100건 id가 든 파일 (있으면 경로, 없으면 빈칸)
import os; os.makedirs(OUT, exist_ok=True); print(os.path.exists(CLEAN))"""),
md("## 함수 불러오기\n아래 두 셀은 고치지 않고 실행만. 키워드는 첫 셀의 `KW` 목록에서 바꿀 수 있음."),
code(rules),
code(steps),
md("## [1] 100건 뽑기\n기간 4구간(2001–05 / 06–10 / 11–15 / 16–21) × 25건, 회사당 최대 3건, seed 42. 지난 시험 기사는 빼고 뽑음."),
code("""x = load_clean(CLEAN)
prev = []
if PREV_TEST:
    p = pd.read_csv(PREV_TEST, dtype=str) if PREV_TEST.endswith('.csv') else pd.read_excel(PREV_TEST, dtype=str)
    prev = p[[c for c in p.columns if c in ('id', '뉴스 식별자')][0]].tolist()
s = draw_100(x, per_period=25, max_per_company=3, seed=42, exclude_ids=prev)
paths = labeling_sheets(s, OUT, names=('예원', '아영'))
# → 두 파일을 예원님, 아영님께 각각 전달 (서로 보지 않고 '정답' 칸 채우기, 헷갈리면 복합=Y)"""),
md("## [3] 일치율 (두 사람 정답을 받은 뒤)\n사람 둘이 다르게 단 기사는 `분류시험_100_최종정답.xlsx`의 '최종' 칸을 영섭이 채우고 이 셀을 다시 실행."),
code("""m = agreement(f'{OUT}/분류시험_100_예원.xlsx', f'{OUT}/분류시험_100_아영.xlsx', classify, OUT)
# 틀린 기사 목록: 분류시험_100_틀린것.xlsx → 키워드를 고친 뒤에는 '새' 100건(seed 다르게)으로 다시 재기"""),
md("## 2-4 감성 점수 — ① 100건으로 점검\nKR-FinBert-SC, 재학습 없음. s = P(긍정) − P(부정). 가장 긍정·부정 10건을 눈으로 보고 방향이 맞는지 확인."),
code("""fb = load_finbert()
s['s'] = score((s.title.fillna('') + ' ' + s.body.fillna('').str[:200]).tolist(), fb)
view = s[['company', 'title', 's']].sort_values('s')
print('가장 부정 10건'); print(view.head(10).to_string(index=False))
print('가장 긍정 10건'); print(view.tail(10).to_string(index=False))"""),
md("## 2-4 감성 점수 — ② 전체 → 회사×월\n모든 기사 분류 → 광고 제외 점수 → 월 배정(마지막 거래일 이후는 다음 달) → 집계. 결과 파일에는 원문이 없음."),
code("""x[['type', '복합', '근거']] = x.apply(lambda r: classify(r.title, r.body), axis=1, result_type='expand')
print(x.type.value_counts(normalize=True).round(3))
use = x[x.type != '광고'].copy()
import time; t0 = time.time()
use['s'] = score((use.title.fillna('') + ' ' + use.body.fillna('').str[:200]).tolist(), fb)
print(f'점수 {len(use):,}건, {time.time()-t0:.0f}초 → 1만 건당 {10000*(time.time()-t0)/max(len(use),1):.0f}초')
x = x.merge(use[['id', 's']], on='id', how='left')
x['month'] = assign_month(x.date)
x[['id', 'code', 'date', 'month', 'type', '복합', 's']].to_csv(f'{OUT}/article_scores_100.csv', index=False, encoding='utf-8-sig')
mon = monthly(x); mon.to_csv(f'{OUT}/news_monthly_100.csv', index=False, encoding='utf-8-sig')
print(mon.shape); print(mon.describe().round(3))"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("영섭_분류시험_감성점수.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
