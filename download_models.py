"""下载说话人分离所需的两个模型到 models/。只需要跑一次。

用法：python download_models.py
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 1_000:
        print(f"已存在，跳过：{dest.name}")
        return
    print(f"下载 {dest.name} ……")
    # Callers pass only the fixed HTTPS model URLs declared in this file.
    urllib.request.urlretrieve(url, dest)  # nosec B310
    print(f"  完成 {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    tarball = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
    fetch(SEGMENTATION_URL, tarball)
    fetch(EMBEDDING_URL, MODELS_DIR / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")

    if not (MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx").exists():
        with tarfile.open(tarball) as archive:
            archive.extractall(MODELS_DIR, filter="data")
        print("解压完成")
    print("模型准备好了，可以在程序里勾选“识别谁在说话”。")


if __name__ == "__main__":
    main()
