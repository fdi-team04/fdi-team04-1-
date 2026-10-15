"""src/ 안의 01 → 05를 번호 순서대로 실행한다. 저장소 맨 위에서 `python run_all.py`."""
import glob
import runpy

for path in sorted(glob.glob("src/0*.py")):
    print("▶", path)
    runpy.run_path(path, run_name="__main__")
