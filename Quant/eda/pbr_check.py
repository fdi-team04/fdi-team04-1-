"""[EDA] PBR 0 구간 점검   담당: 위영섭

데이터 열어보기 분담 3번(11/5): PBR 출처와 값이 0인 구간 확인.
계획 문서 메모: 2000~2001년 전 종목과 2008년까지 일부 월이 0 → 0은 '값 없음'으로 처리 예정.

입력  monthly_pbr_2000_2025.csv (wide: 첫 열 = 월, 나머지 열 = 6자리 종목코드 / 또는 long: month, code, pbr)
출력  <out>/pbr_zero_by_month.csv, pbr_zero_by_year.csv, summary.md

실행  python Quant/eda/pbr_check.py <pbr 파일> [--long <long 파일>] [--out 출력 폴더]

규칙: 2022-01 이후(test) 행은 읽자마자 버린다. 0·음수는 고치지 않고 개수만 센다.
--long 을 주면 배포 데이터 종목과 PBR 종목의 매칭률도 본다.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LAST_TRAIN_MONTH = pd.Period("2021-12", freq="M")


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    lower = {c.lower(): c for c in df.columns}
    code_col = next((lower[k] for k in ["stock_code", "code", "종목코드"] if k in lower), None)
    if code_col is not None:  # 이미 long
        month_col = next(lower[k] for k in ["month", "date", "월", "날짜"] if k in lower)
        val_col = next(c for c in df.columns if c not in (code_col, month_col) and "pbr" in c.lower())
        out = df[[month_col, code_col, val_col]].copy()
        out.columns = ["month", "stock_code", "pbr"]
    else:  # wide: 첫 열이 월
        month_col = df.columns[0]
        out = df.melt(id_vars=month_col, var_name="stock_code", value_name="pbr").rename(columns={month_col: "month"})
    out["month"] = pd.to_datetime(out["month"].astype(str).str.replace("/", "-")).dt.to_period("M")
    out = out[out["month"] <= LAST_TRAIN_MONTH].copy()   # test 차단
    out["stock_code"] = out["stock_code"].astype(str).str.strip().str.extract(r"(\d+)")[0].str.zfill(6)
    out["pbr"] = pd.to_numeric(out["pbr"], errors="coerce")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path)
    ap.add_argument("--long", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("eda_out/pbr"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(a.path, encoding="utf-8-sig", dtype=str)
    p = to_long(raw)
    has = p.dropna(subset=["pbr"])

    by_m = has.groupby("month").agg(
        stocks=("pbr", "size"),
        zero=("pbr", lambda s: int((s == 0).sum())),
        negative=("pbr", lambda s: int((s < 0).sum())),
    )
    by_m["zero_rate"] = by_m["zero"] / by_m["stocks"]
    by_y = by_m.groupby(by_m.index.year)[["stocks", "zero", "negative"]].sum()
    by_y["zero_rate"] = by_y["zero"] / by_y["stocks"]
    all_zero = by_m.index[by_m["zero_rate"] == 1]
    some_zero = by_m.index[(by_m["zero_rate"] > 0) & (by_m["zero_rate"] < 1)]

    by_m.to_csv(a.out / "pbr_zero_by_month.csv", encoding="utf-8-sig")
    by_y.to_csv(a.out / "pbr_zero_by_year.csv", encoding="utf-8-sig")

    lines = [
        "# 데이터 열어보기 3번: PBR 0 구간",
        "",
        f"- 파일: {a.path.name} (원본 열 {raw.shape[1]}개, 2021-12까지만 읽음)",
        f"- 기간: {p['month'].min()} ~ {p['month'].max()}, 값이 있는 칸 {len(has):,} / 전체 {len(p):,}",
        f"- 전 종목이 0인 달: {len(all_zero)}개 ({', '.join(map(str, all_zero[:3]))}{' …' if len(all_zero) > 3 else ''} ~ {all_zero.max() if len(all_zero) else '-'})",
        f"- 일부 종목만 0인 달: {len(some_zero)}개 (마지막 {some_zero.max() if len(some_zero) else '-'})",
        f"- 음수 PBR 칸: {int(by_m['negative'].sum())}개 (자본잠식 가능성 → 1/PBR 계산 전 규칙 필요)",
        "",
        "## 연도별",
        by_y.to_markdown(floatfmt=".3f"),
        "",
        "## 처리 제안",
        "- 0은 '값 없음(결측)'으로 본다. 0으로 두면 1/PBR이 무한대가 된다.",
        "- 결측은 채우지 않고 순위 변환 단계의 '결측 표시 열'로 처리한다(정의서 공통 규칙).",
        "- 전 종목이 0인 기간은 '가치' 변수를 쓸 수 없는 기간으로 보고서에 적는다. 도서관 재무 DB 보완 가능성은 별도 확인.",
    ]
    if a.long is not None:
        lng = pd.read_csv(a.long, encoding="utf-8-sig", dtype={"Stock_Code": str}, usecols=lambda c: c.lower() in ("month", "stock_code", "adjusted_monthly_return"))
        lng.columns = [c.lower() for c in lng.columns]
        lng["month"] = pd.to_datetime(lng["month"]).dt.to_period("M")
        lng = lng[(lng["month"] <= LAST_TRAIN_MONTH) & lng["adjusted_monthly_return"].notna()]
        lng["stock_code"] = lng["stock_code"].astype(str).str.zfill(6)
        m = lng.merge(has[["month", "stock_code", "pbr"]], on=["month", "stock_code"], how="left")
        rate = m.groupby(m["month"].dt.year)["pbr"].apply(lambda s: s.notna().mean()).rename("matched")
        nonzero = m.groupby(m["month"].dt.year)["pbr"].apply(lambda s: (s.fillna(0) != 0).mean()).rename("usable_nonzero")
        mt = pd.concat([rate, nonzero], axis=1)
        mt.to_csv(a.out / "pbr_match_by_year.csv", encoding="utf-8-sig")
        lines += ["", "## 배포 데이터(수익률 있는 종목) 대비 매칭률", mt.to_markdown(floatfmt=".3f")]

    (a.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\n결과: {a.out.resolve()}")


if __name__ == "__main__":
    main()
