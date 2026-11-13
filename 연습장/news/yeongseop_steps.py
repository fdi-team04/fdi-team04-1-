"""영섭 담당: [1] 100건 뽑기, [3] 일치율, 2-4 감성 점수 — 아영 articles_clean 파일 형식에 자동으로 맞춤"""
import os, re, pandas as pd, numpy as np

CAND = {"id": ["id", "nid", "news_id", "뉴스 식별자", "뉴스식별자"],
        "code": ["code", "종목코드", "stock_code", "ticker"],
        "company": ["company", "회사명", "name", "company_name", "회사"],
        "date": ["date", "일자", "날짜"],
        "outlet": ["outlet", "언론사", "press"],
        "title": ["title", "제목"],
        "body": ["body", "본문", "content"]}


def load_clean(path):
    """아영 articles_clean*.csv → 표준 칸(id, code, company, date, outlet, title, body, year)"""
    d = pd.read_csv(path, dtype=str) if path.endswith(".csv") else pd.read_excel(path, dtype=str)
    out = {}
    for k, cs in CAND.items():
        c = next((c for c in cs if c in d.columns), None)
        out[k] = d[c] if c else pd.Series([""] * len(d))
    x = pd.DataFrame(out)
    if (x.id == "").all():
        x["id"] = [f"r{i}" for i in range(len(x))]
    x["code"] = x.code.str.replace(r"\.0$", "", regex=True).str.zfill(6)
    x["date"] = pd.to_datetime(x.date.str[:10].str.replace(r"^(\d{4})(\d{2})(\d{2})$", r"\1-\2-\3", regex=True), errors="coerce")
    x["year"] = x.date.dt.year
    missing = [k for k in ["code", "date", "title"] if (out[k] == "").all()]
    print(f"읽은 줄 {len(x):,} | 칸 연결: " + ", ".join(f"{k}←{next((c for c in CAND[k] if c in d.columns), '없음')}" for k in CAND))
    if missing:
        print("⚠ 꼭 필요한 칸이 없음:", missing)
    return x


PERIODS = [(2001, 2005), (2006, 2010), (2011, 2015), (2016, 2021)]


def draw_100(x, per_period=25, max_per_company=3, seed=42, exclude_ids=()):
    """기간별 25건, 회사당 최대 3건, 시드 고정. exclude_ids = 지난 시험에 쓴 기사 id"""
    pool = x[~x.id.isin(set(exclude_ids)) & x.year.notna()].sample(frac=1, random_state=seed)
    picked, cnt = [], {}
    for a, b in PERIODS:
        p = pool[(pool.year >= a) & (pool.year <= b)]
        n = 0
        for i, r in p.iterrows():
            key = r.code if r.code.strip("0") else r.company
            if cnt.get(key, 0) >= max_per_company:
                continue
            picked.append(i); cnt[key] = cnt.get(key, 0) + 1; n += 1
            if n == per_period:
                break
        if n < per_period:
            print(f"⚠ {a}–{b}: {n}건만 뽑힘 (기사 부족)")
    s = pool.loc[picked].copy()
    s.insert(0, "번호", range(1, len(s) + 1))
    print(f"뽑은 건수 {len(s)} | 기간별 {s.groupby(pd.cut(s.year, [2000, 2005, 2010, 2015, 2021])).size().tolist()} | 회사 수 {s.code.nunique()} | 회사당 최대 {s.code.value_counts().max()}")
    return s


def labeling_sheets(s, out_dir, names=("예원", "아영")):
    """사람 정답지: 컴퓨터 답 없이 2부. 열쇠 파일(id 목록)은 따로 저장"""
    os.makedirs(out_dir, exist_ok=True)
    sheet = s[["번호", "id", "company", "code", "date", "outlet", "title"]].copy()
    sheet["본문 앞 200자"] = s.body.str[:200]
    sheet["date"] = sheet.date.dt.strftime("%Y-%m-%d")
    sheet = sheet.rename(columns={"company": "회사", "code": "종목코드", "date": "날짜", "outlet": "언론사", "title": "제목"})
    sheet["정답 (광고/공시형/논란형/화제형/기타)"] = ""; sheet["복합 (Y/N)"] = ""; sheet["메모"] = ""
    paths = []
    for n in names:
        p = f"{out_dir}/분류시험_100_{n}.xlsx"; sheet.to_excel(p, index=False); paths.append(p)
    s.to_csv(f"{out_dir}/분류시험_100_원본.csv", index=False, encoding="utf-8-sig")
    print("정답지:", paths)
    return paths


VALID = ["광고", "공시형", "논란형", "화제형", "기타"]


def _norm(v):
    v = str(v).strip().replace(" ", "")
    for t in VALID:
        if v.startswith(t[:2]):
            return t
    return np.nan


def agreement(path_a, path_b, classify, out_dir, final_col=None):
    """[3] 사람1·사람2·컴퓨터 비교. 사람 둘이 다른 기사는 '최종' 칸을 영섭이 채우고 다시 실행"""
    col = "정답 (광고/공시형/논란형/화제형/기타)"
    a = pd.read_excel(path_a, dtype=str); b = pd.read_excel(path_b, dtype=str)
    m = a[["번호", "id", "회사", "제목", "본문 앞 200자", col, "복합 (Y/N)"]].merge(
        b[["번호", col, "복합 (Y/N)"]], on="번호", suffixes=("_A", "_B"))
    m["사람A"] = m[f"{col}_A"].map(_norm); m["사람B"] = m[f"{col}_B"].map(_norm)
    hh = (m.사람A == m.사람B).mean()
    m["최종"] = np.where(m.사람A == m.사람B, m.사람A, np.nan)
    fp = f"{out_dir}/분류시험_100_최종정답.xlsx"
    if os.path.exists(fp):                      # 영섭이 채운 최종 정답 반영
        f = pd.read_excel(fp, dtype=str).set_index("번호")["최종"].map(_norm)
        m["최종"] = m.번호.map(f).fillna(m["최종"])
    r = m.apply(lambda x: classify(x["제목"], x["본문 앞 200자"]), axis=1, result_type="expand")
    m[["컴퓨터", "컴퓨터_복합", "근거"]] = r
    ok = m.최종.notna()
    acc = (m.loc[ok, "컴퓨터"] == m.loc[ok, "최종"]).mean()
    tp = m[ok & (m.컴퓨터 == "화제형")]; tr = m[ok & (m.최종 == "화제형")]
    prec = (tp.최종 == "화제형").mean() if len(tp) else np.nan
    rec = (tr.컴퓨터 == "화제형").mean() if len(tr) else np.nan
    print(f"사람끼리 일치 {hh:.0%} ({(m.사람A == m.사람B).sum()}/{len(m)})")
    print(f"최종 정답이 정해진 기사 {ok.sum()}건 | 컴퓨터 일치율 {acc:.0%} (목표 80%) | 화제형 정밀도 {prec:.0%}, 재현율 {rec:.0%}")
    print(pd.crosstab(m.loc[ok, "최종"], m.loc[ok, "컴퓨터"], rownames=["사람 최종"], colnames=["컴퓨터"], margins=True))
    if not os.path.exists(fp):
        m[["번호", "회사", "제목", "사람A", "사람B", "최종"]].to_excel(fp, index=False)
        print(f"→ 사람 둘이 다른 {(~ok).sum()}건: {fp} 의 '최종' 칸을 채우고 이 셀을 다시 실행")
    m[ok & (m.컴퓨터 != m.최종)][["번호", "회사", "제목", "최종", "컴퓨터", "근거"]].to_excel(f"{out_dir}/분류시험_100_틀린것.xlsx", index=False)
    return m


# ---------- 2-4 감성 점수 ----------
def load_finbert(model="snunlp/KR-FinBert-SC"):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForSequenceClassification.from_pretrained(model).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; mdl.to(dev)
    if dev == "cuda": mdl.half()
    lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
    print("모델", model, "| 장치", dev, "| 라벨", mdl.config.id2label)
    return tok, mdl, dev, lab


def score(texts, fb, bs=128, max_len=160):
    """s = P(긍정) − P(부정), −1~+1. 재학습 없음"""
    import torch
    tok, mdl, dev, lab = fb; out = []
    for i in range(0, len(texts), bs):
        enc = tok(list(texts[i:i + bs]), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.no_grad():
            p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
        out.append(p[:, lab["positive"]] - p[:, lab["negative"]])
    return np.concatenate(out) if out else np.array([])


def assign_month(dates):
    """마지막 거래일(평일 근사) 당일·이후 기사는 다음 달로"""
    per = dates.dt.to_period("M")
    last = (dates + pd.offsets.BMonthEnd(0)).dt.normalize()
    return per.where(dates < last, per + 1).astype(str)


def monthly(x):
    """회사×월 집계 (원문 없음 → 밖으로 내보내도 됨)"""
    g = x.groupby(["month", "code"])
    out = g.size().rename("n_all").to_frame()
    for t, k in [("화제형", "topic"), ("공시형", "disc"), ("논란형", "contro"), ("광고", "ad")]:
        sub = x[x.type == t].groupby(["month", "code"])
        out[f"n_{k}"] = sub.size(); out[f"{k}_tone"] = sub.s.mean()
    for c in [c for c in out if c.startswith("n_")]:
        out[c] = out[c].fillna(0).astype(int)
    return out.reset_index()
