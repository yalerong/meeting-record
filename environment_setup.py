"""Background environment preparation; callbacks receive text, never Tk objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from urllib import request

from console_support import AppPaths

Progress = Callable[[str], None]
Cancel = Callable[[], bool]
WHISPER_MODEL = "large-v3-turbo"
WHISPER_FILES = ("model.bin", "config.json", "tokenizer.json", "vocabulary.json")


class SetupCancelled(RuntimeError):
    pass


def check_cancel(cancel: Cancel | None) -> None:
    if cancel is not None and cancel():
        raise SetupCancelled("已取消环境准备，已下载的完整文件会保留。")


def whisper_model_dir(paths: AppPaths) -> Path:
    return paths.models_dir / "whisper-large-v3-turbo"


def whisper_ready(path: Path) -> bool:
    """Check local completeness, without loading weights or contacting the Hub.

    The official model repository supplies these four files. In faster-whisper,
    a missing tokenizer would otherwise fall back to an online pretrained tokenizer.
    This does not validate every byte of the weight file.
    """
    path = Path(path)
    try:
        if not all((path / name).is_file() and (path / name).stat().st_size > 0
                   for name in WHISPER_FILES):
            return False
        for name in WHISPER_FILES[1:]:
            with (path / name).open(encoding="utf-8") as source:
                data = json.load(source)
            expected = list if name == "vocabulary.json" else dict
            if not isinstance(data, expected) or not data:
                return False
        return True
    except (OSError, ValueError):
        return False


def prepare_whisper(
    path: Path, progress: Progress | None = None, cancel: Cancel | None = None,
) -> Path:
    path = Path(path)
    report = progress or (lambda message: None)
    check_cancel(cancel)
    if whisper_ready(path):
        report("转写模型已就绪，无需重复下载。")
        return path
    from faster_whisper.utils import download_model

    path.mkdir(parents=True, exist_ok=True)
    report("正在下载转写模型（约 1.6 GB）；下载器不提供逐字节进度，取消将在当前下载返回后生效。")
    # The official API delegates atomic file handling/resume to huggingface_hub.
    download_model(WHISPER_MODEL, output_dir=str(path))
    check_cancel(cancel)
    if not whisper_ready(path):
        raise RuntimeError("转写模型下载未完整完成，请重试；缺少或损坏了本地模型文件。")
    report("转写模型下载完成。")
    return path


def ollama_pull(
    model: str, progress: Progress | None = None, cancel: Cancel | None = None,
) -> None:
    report = progress or (lambda message: None)
    check_cancel(cancel)
    req = request.Request(
        "http://127.0.0.1:11434/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    report(f"正在准备纪要模型 {model}；取消在收到下一条进度或网络超时后生效。")
    # Read timeout bounds a stalled stream; normal downloads keep reporting events.
    with request.urlopen(req, timeout=60) as response:  # nosec B310: fixed loopback endpoint
        for line in response:
            check_cancel(cancel)
            if not line.strip():
                continue
            event = json.loads(line)
            if not isinstance(event, dict):
                raise RuntimeError("Ollama 返回了无效进度。")
            if event.get("error"):
                raise RuntimeError(f"Ollama 下载失败：{event['error']}")
            status = str(event.get("status", ""))
            total, completed = event.get("total"), event.get("completed")
            if isinstance(total, int) and total > 0 and isinstance(completed, int):
                report(f"{model}：{status} {min(100, completed * 100 // total)}%")
            elif status:
                report(f"{model}：{status}")
            if status == "success":
                check_cancel(cancel)
                return
    check_cancel(cancel)
    raise RuntimeError("Ollama 下载连接提前结束，未收到成功确认；请重试。")
