"""run_all.py — src/01 → 05를 순서대로 실행한다.

사용:  python run_all.py            # 기본: train 기간만 (test 파일은 열지 않는다)
       python run_all.py --final    # 판단을 모두 고정한 뒤 test 예측을 한 번 산출
각 단계는 앞 단계가 data/processed에 저장한 파일만 읽으므로 어느 단계든 따로 실행해도 된다.
실행 기록(설정·패키지 버전·단계별 결과)은 logs/ 에 남긴다.
"""
import argparse
import os
import platform
import random
import runpy
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_load_clean",
    "02_features",
    "03_text_features",
    "04_model",
    "05_evaluate",
]
PACKAGES = ["pandas", "numpy", "scipy", "statsmodels", "pyyaml"]


def package_versions() -> list[str]:
    """실행 기록용: 주요 패키지 버전 목록(설치 안 된 패키지는 표시만 한다)."""
    lines = []
    for name in PACKAGES:
        try:
            lines.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"{name} (not installed)")
    return lines


def set_seed(seed: int) -> None:
    """난수 시드를 config.yaml 값 하나로 고정한다."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="01~05 단계를 순서대로 실행한다.")
    parser.add_argument(
        "--final",
        action="store_true",
        help="test(2022~2025) 예측을 한 번 산출한다. 판단을 모두 고정한 뒤에만 쓴다.",
    )
    args = parser.parse_args()

    # 각 파일이 config.yaml을 상대 경로로 읽으므로 저장소 루트에서 실행한다.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

    set_seed(int(cfg.get("seed", 0)))
    # 하위 단계가 test 파일을 열어도 되는지 이 값으로만 판단한다(기본: 열지 않음).
    os.environ["FDI_FINAL"] = "1" if args.final else "0"

    Path("logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"run_{stamp}.log"
    log = [
        f"start      : {datetime.now().isoformat(timespec='seconds')}",
        f"mode       : {'FINAL (test 예측 1회)' if args.final else 'TRAIN ONLY'}",
        f"seed       : {cfg.get('seed', 0)}",
        f"python     : {platform.python_version()}",
        "packages   : " + ", ".join(package_versions()),
    ]

    for step in STEPS:
        path = ROOT / "src" / f"{step}.py"
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없습니다. 파일 이름이 부록 B와 같은지 확인하세요.")
        print(f"[run_all] {step} 실행")
        # 파일 이름이 숫자로 시작해 import할 수 없으므로 경로로 실행한다.
        runpy.run_path(str(path), run_name="__main__")
        log.append(f"done       : {step}")

    log.append(f"end        : {datetime.now().isoformat(timespec='seconds')}")
    log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"[run_all] 완료. 기록: {log_path}")


if __name__ == "__main__":
    main()
