"""下载说话人分离模型到用户的持久模型目录。只需要跑一次。

用法：python download_models.py
"""

from __future__ import annotations

import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from console_support import get_app_paths
from environment_setup import Cancel, Progress, check_cancel

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
)


def fetch(
    url: str, dest: Path, progress: Progress | None = None, cancel: Cancel | None = None,
) -> None:
    if url not in (SEGMENTATION_URL, EMBEDDING_URL):
        raise ValueError("只允许下载预设的官方模型 URL。")
    report = progress or (lambda message: None)
    check_cancel(cancel)
    if dest.exists() and dest.stat().st_size > 1_000:
        report(f"已存在，跳过：{dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    report(f"下载 {dest.name} ……")
    try:
        # Only fixed HTTPS URLs above are accepted; publish a file only after EOF.
        with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
            expected = int(response.headers.get("Content-Length", "0"))
            received = 0
            with partial.open("wb") as output:
                while True:
                    check_cancel(cancel)
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    received += len(block)
                    suffix = f" {min(100, received * 100 // expected)}%" if expected else ""
                    report(f"{dest.name}：{received / 1e6:.1f} MB{suffix}")
                output.flush()
                os.fsync(output.fileno())
        check_cancel(cancel)
        if received <= 1_000 or (expected and received != expected):
            raise RuntimeError(f"模型下载不完整：{dest.name}，请重试。")
        os.replace(partial, dest)
    finally:
        partial.unlink(missing_ok=True)
    report(f"下载完成：{dest.name}")


def prepare_models(
    models_dir: Path, progress: Progress | None = None, cancel: Cancel | None = None,
) -> None:
    models_dir = Path(models_dir)
    report = progress or (lambda message: None)
    check_cancel(cancel)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_name = "sherpa-onnx-pyannote-segmentation-3-0"
    segmentation = models_dir / model_name / "model.onnx"
    if not segmentation.is_file() or segmentation.stat().st_size <= 1_000:
        tarball = models_dir / f"{model_name}.tar.bz2"
        fetch(SEGMENTATION_URL, tarball, report, cancel)
        report("正在解压说话人分离模型；取消将在当前解压完成后生效。")
        with tempfile.TemporaryDirectory(prefix=".segmentation-", dir=models_dir) as temporary:
            with tarfile.open(tarball) as archive:
                archive.extractall(temporary, filter="data")
            check_cancel(cancel)
            extracted = Path(temporary) / model_name / "model.onnx"
            if not extracted.is_file() or extracted.stat().st_size <= 1_000:
                raise RuntimeError("下载的分离模型包缺少完整 model.onnx。")
            segmentation.parent.mkdir(parents=True, exist_ok=True)
            os.replace(extracted, segmentation)

    fetch(
        EMBEDDING_URL, models_dir / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
        report, cancel,
    )
    check_cancel(cancel)
    report("说话人模型准备完成，可以勾选“识别谁在说话”。")


def main() -> None:
    prepare_models(get_app_paths().models_dir, progress=print)


if __name__ == "__main__":
    main()
