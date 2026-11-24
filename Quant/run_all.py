"""src/의 단계를 번호 순서대로 실행한다. Quant 폴더에서 `python run_all.py`.

기본    : 04 → 06 (만들어진 입력 파일 → 결과. 채점용 재현)
--all   : 01 → 06 전부 (원자료·외부 데이터·기사가 있는 곳에서 입력 파일부터 다시 만들 때)
--final : test 행까지 예측하고 prediction.csv를 만든다 (test는 한 번만). --all과 같이 써도 된다
"""
import glob
import runpy
import sys

steps = sorted(glob.glob("src/0*.py"))  # 파일 이름 순서 = 실행 순서
if "--all" not in sys.argv:
    steps = [p for p in steps if p >= "src/04"]  # 04 이후만
for path in steps:
    print("▶", path)
    runpy.run_path(path, run_name="__main__")
