"""기사 정리(메모리 절약판): 엑셀을 한 파일씩 읽으면서 바로 걸러서, 기사가 1,000만 건이 넘어도 메모리가 모자라지 않게 한다.
사용법:  python run_clean3.py 기사폴더 [출력폴더]   (run_clean2 의 개선판: 파일 하나가 깨져도 멈추지 않고, 250개마다 중간 저장해서 끊겨도 이어서 가능)
  파일마다 ① 분석제외·날짜 없음 제거 ② 제목에 회사 이름(옛 이름 포함)이 있는 기사만 남김 ③ 중복(제목+언론사+일자) 제거
  를 하고, 남은 기사를 출력폴더의 deduped_names.pkl 에 저장한다.
  회사 목록의 기간(first_month, last_month)만 고친 뒤 다시 돌리면 이 파일을 읽어서 빨리 끝난다(엑셀을 안 읽는다).
  회사 목록에 새 이름을 넣었다면 deduped_names.pkl 을 지우고 다시 돌린다.
  (03_text_features.py, company_list.csv 는 이 파일과 같은 폴더에 있어야 한다)
주의: 결과 파일에는 기사 제목·본문이 들어 있으니 깃에 올리지 말고 드라이브에만 둔다.
월말 기준일은 '평일'로 임시 계산한다(공휴일 반영 전)."""
import sys, os, re, gc, glob, warnings, importlib.util
import pandas as pd

raw_dir = os.path.abspath(sys.argv[1])
out = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else "out")
here = os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
if not os.path.exists("config.yaml"):
    open("config.yaml", "w").write("data_dir: ./data\ntext_model: snunlp/KR-FinBert-SC\n")
spec = importlib.util.spec_from_file_location("t03", "03_text_features.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
os.makedirs(out, exist_ok=True)
nd = m.load_name_dict("company_list.csv")

COLS = {"일자": "date", "언론사": "outlet", "제목": "title", "본문": "body", "분석제외 여부": "excluded_flag"}


BODY_MAX = int(os.environ.get("BODY_MAX", "300"))   # 본문은 앞 300자만 보관(메모리 절약). 전체 본문은 드라이브 원본 엑셀에 있음
CHUNK = int(os.environ.get("CHUNK","250"))                                          # 250개 파일마다 중간 저장(꺼져도 이어서 가능)


_G = {}


def _work(f):
    try:
        return f, read_one(f, _G["pat"], _G["engine"]), None
    except Exception as e:
        return f, (None, 0), repr(e)[:150]


def read_one(f, pat, engine):
    d = pd.read_excel(f, usecols=lambda c: c in COLS, engine=engine).rename(columns=COLS)
    if "title" not in d.columns or "date" not in d.columns:
        return None, 0
    for c in COLS.values():
        if c not in d.columns:
            d[c] = pd.NA
    n = len(d)
    d["date"] = pd.to_datetime(d["date"].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    flag = d["excluded_flag"].fillna("").astype(str)
    d = d[(~flag.str.contains("중복|예외")) & d["date"].notna()]
    t = d["title"].fillna("").astype(str).str.replace("車", "차")
    has = t.str.contains(pat)
    d = d[has].copy(); d["title"] = t[has]
    d["body"] = d["body"].fillna("").astype(str).str.slice(0, BODY_MAX).str.replace("車", "차")
    return d[["date", "outlet", "title", "body"]], n


def stream_load(raw_dir, nd):
    names = sorted({n.strip().replace("車", "차") for v in nd.values() for (n, _, _) in v
                    if isinstance(n, str) and len(n.strip()) >= 2}, key=len, reverse=True)
    pat = re.compile("|".join(re.escape(n) for n in names))
    try:
        import python_calamine  # noqa: F401
        engine = "calamine"
    except ImportError:
        engine = None
        print("[주의] python-calamine 이 없어서 느린 방식으로 읽어요. !pip install python-calamine 을 먼저 하세요", flush=True)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.xlsx")))
    import multiprocessing as mp
    _G["pat"], _G["engine"] = pat, engine
    pool = mp.get_context("fork").Pool(max(1, min(os.cpu_count() or 1, int(os.environ.get("WORKERS", "2")))))
    pdir = f"{out}/parts"; os.makedirs(pdir, exist_ok=True)
    n_read = 0; failed = []; skipped = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ci in range(0, len(files), CHUNK):
            pf = f"{pdir}/part_{ci // CHUNK:04d}_of_{len(files)}.pkl"
            if os.path.exists(pf):
                print(f"[stream] 파일 {ci + 1}~{min(ci + CHUNK, len(files))}: 저장된 중간 결과 사용", flush=True)
                continue
            parts = []
            batch = files[ci:ci + CHUNK]
            for k, (f, res, err) in enumerate(pool.imap(_work, batch, chunksize=2), ci + 1):
                d, n = res
                if err:
                    failed.append((os.path.basename(f), err)); print("  (읽기 실패, 건너뜀):", os.path.basename(f), err[:100], flush=True); continue
                if d is None:
                    skipped.append(os.path.basename(f)); print("  (기사 열이 없어 건너뜀):", os.path.basename(f), flush=True); continue
                n_read += n; parts.append(d)
                if k % 25 == 0 or k == len(files):
                    print(f"[stream] {k}/{len(files)} 파일 / 이 실행에서 읽음 {n_read:,}", flush=True)
            pd.concat(parts, ignore_index=True).to_pickle(pf) if parts else pd.DataFrame(columns=["date", "outlet", "title", "body"]).to_pickle(pf)
            del parts; gc.collect()
    pool.close(); pool.join()
    if failed or skipped:
        with open(f"{out}/skipped_files.txt", "w", encoding="utf-8") as fh:
            fh.write("기사 열 없음:\n" + "\n".join(skipped) + "\n\n읽기 실패:\n" + "\n".join(f"{a}\t{b}" for a, b in failed))
        print(f"[stream] 건너뛴 파일 {len(skipped)}개 / 읽기 실패 {len(failed)}개 → {out}/skipped_files.txt", flush=True)
    df = pd.concat([pd.read_pickle(p) for p in sorted(glob.glob(f"{pdir}/part_*_of_{len(files)}.pkl"))], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(["title", "outlet", "date"]).reset_index(drop=True)
    print(f"[stream] 합친 기사 {before:,} → 중복 제거 후 {len(df):,}", flush=True)
    return df


pk = f"{out}/deduped_names.pkl"
if os.path.exists(pk):
    print("[run_clean2] 저장된 deduped_names.pkl 을 읽어요(엑셀 읽기 건너뜀)", flush=True)
    dd = pd.read_pickle(pk)
else:
    dd = stream_load(raw_dir, nd)
    dd.to_pickle(pk)
    print("[run_clean2] deduped_names.pkl 저장", flush=True)
cl = m.clean_news(dd, nd, already_deduped=True); del dd; gc.collect()
print("정리 후 기사:", len(cl))
d = pd.to_datetime(cl["date"])
td = pd.bdate_range(d.min() - pd.offsets.MonthBegin(1), d.max() + pd.offsets.MonthEnd(1))
am = m.assign_month(cl, td)
am.to_csv(f"{out}/articles_clean.csv", index=False, encoding="utf-8-sig")
cnt = am.groupby(["code", "month"]).size().rename("n_articles").reset_index()
cnt.to_csv(f"{out}/article_counts_code_month.csv", index=False, encoding="utf-8-sig")
tot = cnt.groupby("code").n_articles.sum().sort_values(ascending=False)
print("기사가 붙은 종목 수:", cnt.code.nunique()); print(tot.head(10).to_string())
print("저장:", f"{out}/articles_clean.csv", f"{out}/article_counts_code_month.csv")
