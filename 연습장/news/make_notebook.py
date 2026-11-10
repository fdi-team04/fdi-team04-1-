"""Colab 노트북 생성: 4팀_뉴스점수_Colab.ipynb"""
import json
core = open("news_core.py", encoding="utf-8").read()

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

cells = [
md("""# 4팀 뉴스 점수 만들기 (H3·H4용) — Colab에서 실행

**하는 일:** 드라이브의 빅카인즈 기사(담당 1~4 폴더) → 정리 → 제목으로 회사 매칭 → 유형 분류(공시형·논란형·화제형·기타)
→ **KR-FinBert-SC 감성 점수** → 종목×월 요약 파일 1개.

**결과로 받는 것 (드라이브 `내 드라이브/4팀_뉴스결과/`)**
- `news_monthly.csv` ← **이것만 Claude 채팅에 올리면 됨** (종목·월별 기사 수와 평균 점수, 원문 없음)
- `news_coverage.csv` ← 종목·연도별 기사 수 (수집 누락 확인용, 같이 올리기)
- `type_check_sample200.csv` ← 사람 확인용 무작위 200건(제목 포함 → **드라이브 밖으로 내보내지 말 것**)

**시작 전 준비 (1번만)**
1. 런타임 → 런타임 유형 변경 → **T4 GPU**
2. 공유받은 폴더 `뉴스_원몬_비공개`가 내 드라이브에 없으면: 드라이브 웹에서 폴더 오른쪽 클릭 → **정리 → 바로가기 추가 → 내 드라이브**
3. 위에서부터 셀을 차례로 실행 (중간에 끊겨도 다시 실행하면 이어서 함: 파일별·점수별 캐시)

이용 조건: 모델은 재학습하지 않고 그대로 사용, 원문은 드라이브 밖으로 내보내지 않음."""),
code("""from google.colab import drive
drive.mount('/content/drive')
!pip -q install python-calamine
import os, glob
# 폴더 위치 자동 찾기 (못 찾으면 아래 NEWS_DIR을 직접 고치기)
cands = glob.glob('/content/drive/MyDrive/**/뉴스_원몬_비공개', recursive=True) + glob.glob('/content/drive/Shareddrives/**/뉴스_원몬_비공개', recursive=True)
NEWS_DIR = cands[0] if cands else '/content/drive/MyDrive/뉴스_원몬_비공개'
OUT_DIR = '/content/drive/MyDrive/4팀_뉴스결과'
CACHE = OUT_DIR + '/cache'
os.makedirs(CACHE, exist_ok=True)
print('뉴스 폴더:', NEWS_DIR, os.path.exists(NEWS_DIR))
files = sorted(glob.glob(NEWS_DIR + '/**/*.xlsx', recursive=True))
SEARCH = [f for f in files if '검색목록' in os.path.basename(f)]
files = [f for f in files if '검색목록' not in os.path.basename(f)]
print('기사 파일', len(files), '개 /', round(sum(os.path.getsize(f) for f in files)/1e9, 2), 'GB')
print('검색목록 파일:', SEARCH)
import collections; print(collections.Counter(f.split('/')[-2] for f in files))"""),
md("## 1. 처리 함수 (팀 저장소 `연습장/news/news_core.py`와 같은 코드)"),
code(core),
md("## 2. 기사 읽기 (파일마다 캐시 → 끊겨도 이어서)"),
code("""import pandas as pd, numpy as np, hashlib, time
names = load_names(SEARCH[0]); match = build_matcher(names)
print('회사 이름', len(names), '개, 종목', names.code.nunique(), '개')
parts = []; t0 = time.time(); n_read = 0
for i, f in enumerate(files):
    key = hashlib.md5((f + str(os.path.getsize(f))).encode()).hexdigest()[:12]
    cp = f'{CACHE}/m_{key}.parquet'
    if os.path.exists(cp):
        d = pd.read_parquet(cp)
    else:
        d = read_one(f)
        d['n_file'] = len(d)
        d['codes'] = d.title.fillna('').str.replace('車', '차').map(match)
        d = d[d.codes.str.len() > 0]          # 제목에 회사 이름이 있는 기사만 남겨 메모리 절약
        d.to_parquet(cp, index=False)
    parts.append(d)
    if i % 10 == 0: print(i, '/', len(files), f.split('/')[-2:], len(d), f'{time.time()-t0:.0f}s')
raw = pd.concat(parts, ignore_index=True)
art, n0 = clean(raw)
print(f'제목에 회사 이름이 있는 기사 {n0:,} → 분석제외·언론사·중복 제거 후 {len(art):,}')"""),
md("## 3. 제목으로 회사 매칭 + 유형 분류 + 월 배정"),
code("""m = art.explode('codes').dropna(subset=['codes']).rename(columns={'codes': 'code'})
m = m[(m.date >= '2000-12-01') & (m.date <= '2021-12-31')]
m['type'] = m.title.map(classify)
m['month'] = assign_month(m.date)
m = m[m.month <= '2021-12']
print(f'제목에 회사 이름이 있는 기사(종목별) {len(m):,}건, 종목 {m.code.nunique()}개')
print(m.type.value_counts(normalize=True).round(3))
# 사람 확인용 무작위 200건 (교수님 피드백 ②: AI 분류를 사람이 확인)
smp = m.sample(200, random_state=1109)[['code', 'date', 'outlet', 'title', 'type']].assign(사람분류_1='', 사람분류_2='', 메모='')
smp.to_csv(OUT_DIR + '/type_check_sample200.csv', index=False, encoding='utf-8-sig')"""),
md("## 4. KR-FinBert-SC 감성 점수 (GPU, 기사별 캐시)\n점수 s = P(긍정) − P(부정), −1~+1. 제목 + 본문 200자를 넣는다. 광고는 점수를 매기지 않는다."),
code("""import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
MODEL = 'snunlp/KR-FinBert-SC'
tok = AutoTokenizer.from_pretrained(MODEL)
mdl = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; mdl.to(dev)
if dev == 'cuda': mdl.half()
lab = {v.lower(): k for k, v in mdl.config.id2label.items()}; print(dev, mdl.config.id2label)
SC = OUT_DIR + '/scores_by_nid.parquet'
done = pd.read_parquet(SC) if os.path.exists(SC) else pd.DataFrame(columns=['nid', 's'])
todo = m[(m.type != '광고')].drop_duplicates('nid')
todo = todo[~todo.nid.isin(done.nid)]
print('점수 매길 기사', len(todo), '건 (이미 있음', len(done), ')')
texts = (todo.title + ' ' + todo.body.str[:200]).tolist(); ids = todo.nid.tolist()
BS = 256; buf = []; t0 = time.time()
for i in range(0, len(texts), BS):
    enc = tok(texts[i:i+BS], padding=True, truncation=True, max_length=160, return_tensors='pt').to(dev)
    with torch.no_grad():
        p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
    buf.append(pd.DataFrame({'nid': ids[i:i+BS], 's': p[:, lab['positive']] - p[:, lab['negative']]}))
    if (i // BS) % 200 == 199 or i + BS >= len(texts):
        done = pd.concat([done] + buf, ignore_index=True); buf = []; done.to_parquet(SC, index=False)
        print(f'{i+BS:,}/{len(texts):,}  {time.time()-t0:.0f}s')"""),
md("## 5. 종목×월 요약 저장 (원문 없음 → 채팅에 올려도 됨)"),
code("""sc = pd.read_parquet(SC)
scored = m.merge(sc, on='nid', how='left')
mon = aggregate(scored)
mon.to_csv(OUT_DIR + '/news_monthly.csv', index=False, encoding='utf-8-sig')
cov = scored.assign(year=scored.month.str[:4]).groupby(['code', 'year']).size().unstack(fill_value=0)
cov.to_csv(OUT_DIR + '/news_coverage.csv', encoding='utf-8-sig')
print(mon.shape, '종목', mon.code.nunique(), '개월', mon.month.nunique())
print(mon[['n_all','n_topic','n_disc','n_contro','topic_tone','all_tone']].describe().round(3))
print('저장 위치:', OUT_DIR)
print('→ news_monthly.csv, news_coverage.csv 두 파일을 Claude 채팅에 올려 주세요.')"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
json.dump(nb, open("4팀_뉴스점수_Colab.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("ok")
