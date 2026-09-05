from __future__ import annotations

import collections
import ctypes
import gc
import json
import math
import os
import queue
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from urllib import request

import sounddevice as sd


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "meetings"
MODELS_DIR = APP_DIR / "models"
SEGMENTATION_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMBEDDING_MODEL = MODELS_DIR / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2
BLOCK_SIZE = 1_600
WHISPER_MODEL = "large-v3-turbo"
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_CHUNK_MODEL = "qwen2.5:3b"

SILENCE_PEAK = 60
BOOST_TRIGGER_PEAK = 8_000
BOOST_TARGET_PEAK = 26_000
MAX_BOOST = 20.0
SILENCE_WARN_SEC = 15.0
STALL_RESTART_SEC = 2.0
STALL_WARN_SEC = 3.0
FLUSH_INTERVAL_SEC = 5.0
PREROLL_SEC = 30.0
MIN_FREE_BYTES = 2 * 1024 ** 3
MAX_WAV_DATA_BYTES = 0xFFFF_FFFF - 36  # RIFF sizes are 32-bit
SLEEP_REASSERT_SEC = 30.0
DIARIZE_THRESHOLD = 0.5

ES_CONTINUOUS = 0x8000_0000
ES_SYSTEM_REQUIRED = 0x0000_0001
ES_DISPLAY_REQUIRED = 0x0000_0002

_kernel32 = ctypes.windll.kernel32
_kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint]
_kernel32.SetThreadExecutionState.restype = ctypes.c_uint


def prevent_system_sleep(enabled: bool) -> bool:
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if enabled else 0)
    return _kernel32.SetThreadExecutionState(flags) != 0


def safe_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "-", value.strip())
    return cleaned[:50].strip(" .-") or "会议"


def timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def block_peak(data: bytes) -> int:
    count = len(data) // SAMPLE_WIDTH
    if count == 0:
        return 0
    samples = struct.unpack(f"<{count}h", data[: count * SAMPLE_WIDTH])
    return max(max(samples), -min(samples))


def peak_to_dbfs(peak: int) -> float:
    if peak <= 0:
        return -60.0
    return max(-60.0, 20 * math.log10(min(peak, 32_767) / 32_768))


def level_percent(peak: int) -> float:
    return max(0.0, min(100.0, (peak_to_dbfs(peak) + 60.0) / 60.0 * 100.0))


SYSTEM_AUDIO_NAME_HINTS = (
    "立体声混音",
    "电脑扬声器",
    "stereo mix",
    "what u hear",
    "wave out mix",
    "waveout mix",
    "mixed output",
    "loopback",
    "monitor of",
    "computer speakers",
)


def is_system_audio_input(name: str) -> bool:
    normalized = name.casefold()
    return any(hint in normalized for hint in SYSTEM_AUDIO_NAME_HINTS)


def system_audio_priority(name: str) -> tuple[bool, int]:
    normalized = name.casefold()
    secondary = any(hint in normalized for hint in ("2nd", "secondary", "耳机", "headphone"))
    direct_loopback = any(
        hint in normalized for hint in ("电脑扬声器", "computer speakers", "loopback", "monitor of")
    )
    stereo_mix = any(
        hint in normalized for hint in ("立体声混音", "stereo mix", "what u hear", "wave out mix")
    )
    category = 0 if direct_loopback else 1 if stereo_mix else 2
    return secondary, category


def select_input_format(
    default_samplerate: float,
    max_input_channels: int,
    supports,
) -> tuple[int, int]:
    native_rate = int(default_samplerate)
    candidates = [(SAMPLE_RATE, 1), (native_rate, 1), (native_rate, min(2, max_input_channels))]
    attempted: set[tuple[int, int]] = set()
    for candidate in candidates:
        if candidate in attempted or candidate[1] <= 0:
            continue
        attempted.add(candidate)
        if supports(*candidate):
            return candidate
    raise RuntimeError("设备不支持可用的 PCM 录音格式")


def downmix_to_mono(data: bytes, channels: int) -> bytes:
    if channels <= 1:
        return data
    import numpy

    samples = numpy.frombuffer(data, dtype="<i2").reshape(-1, channels)
    return samples.mean(axis=1).astype("<i2").tobytes()


def _resampled_mono_block(
    handle,
    output_start: int,
    count: int,
    output_rate: int,
    start_offset: float = 0.0,
):
    import numpy

    source_rate = handle.getframerate()
    source_frames = handle.getnframes()
    source_channels = handle.getnchannels()
    positions = (
        output_start
        + numpy.arange(count, dtype=numpy.float64)
        - start_offset * output_rate
    ) * source_rate / output_rate
    valid = (positions >= 0) & (positions < source_frames)
    result: Any = numpy.zeros(count, dtype=numpy.float64)
    if not valid.any():
        return result

    valid_positions = positions[valid]
    source_start = int(numpy.floor(valid_positions[0]))
    source_stop = min(source_frames, int(numpy.floor(valid_positions[-1])) + 2)
    handle.setpos(source_start)
    raw = handle.readframes(source_stop - source_start)
    samples = numpy.frombuffer(raw, dtype="<i2").reshape(-1, source_channels)
    mono = samples.astype(numpy.float64).mean(axis=1)
    source_axis: Any = numpy.arange(source_start, source_start + len(mono), dtype=numpy.float64)
    result[valid] = numpy.interp(valid_positions, source_axis, mono)
    return result


def mix_wav_tracks(
    paths: list[Path],
    destination: Path,
    output_rate: int = SAMPLE_RATE,
    start_offsets: list[float] | None = None,
) -> None:
    import numpy

    if not paths:
        raise ValueError("至少需要一条音轨")
    offsets = [0.0] * len(paths) if start_offsets is None else start_offsets
    if len(offsets) != len(paths) or any(offset < 0 for offset in offsets):
        raise ValueError("每条音轨必须有一个非负起始偏移")
    handles = [wave.open(str(path)) for path in paths]
    writer: WavWriter | None = None
    try:
        for handle in handles:
            if handle.getsampwidth() != SAMPLE_WIDTH or handle.getcomptype() != "NONE":
                raise ValueError("只能混合 16 位 PCM WAV")
        total_frames = max(
            math.ceil(
                offset * output_rate
                + handle.getnframes() * output_rate / handle.getframerate()
            )
            for handle, offset in zip(handles, offsets, strict=True)
        )
        writer = WavWriter(destination, samplerate=output_rate, channels=1)
        chunk_frames = output_rate * 10
        for start in range(0, total_frames, chunk_frames):
            count = min(chunk_frames, total_frames - start)
            mixed = numpy.zeros(count, dtype=numpy.float64)
            for handle, offset in zip(handles, offsets, strict=True):
                mixed += _resampled_mono_block(
                    handle,
                    start,
                    count,
                    output_rate,
                    start_offset=offset,
                )
            # Plain sum: any per-sample gain switching distorts speech, and a constant
            # halving would quiet whichever side is speaking alone. Clipping only
            # happens when both sides are loud at the same instant.
            numpy.clip(mixed, -32_768, 32_767, out=mixed)
            writer.write(mixed.astype("<i2").tobytes())
    finally:
        for handle in handles:
            handle.close()
        if writer is not None:
            writer.close()


def wav_stats(path: Path) -> tuple[float, int]:
    with wave.open(str(path)) as handle:
        rate = handle.getframerate() or SAMPLE_RATE
        frames = handle.getnframes()
        peak = 0
        while True:
            data = handle.readframes(rate * 30)
            if not data:
                break
            peak = max(peak, block_peak(data))
    return frames / rate, peak


def verify_wav_file(path: Path, expected_seconds: float = 0.0) -> float:
    # Header only: recorders already track the peak, and scanning multi-GB tracks
    # would freeze the UI thread.
    with wave.open(str(path)) as handle:
        duration = handle.getnframes() / (handle.getframerate() or SAMPLE_RATE)
    if duration <= 0:
        raise RuntimeError(f"录音文件没有音频数据：{path.name}")
    if expected_seconds > 0:
        allowed_gap = max(2.0, expected_seconds * 0.05)
        if duration < max(0.1, expected_seconds - allowed_gap):
            raise RuntimeError(
                f"录音时长异常：{path.name} 只有 {duration:.1f} 秒，程序记录了 {expected_seconds:.1f} 秒"
            )
    return duration


def diarization_source(path: Path, dest: Path) -> Path:
    """Return ``path`` if it is already 16 kHz mono, otherwise write such a copy to ``dest``."""
    with wave.open(str(path)) as handle:
        if handle.getframerate() == SAMPLE_RATE and handle.getnchannels() == 1:
            return path
    mix_wav_tracks([path], dest)
    return dest


def boost_wav(src: Path, dst: Path, peak: int) -> float:
    import numpy

    if peak <= 0:
        return 1.0
    gain = min(MAX_BOOST, BOOST_TARGET_PEAK / peak)
    if gain <= 1.05:
        return 1.0
    with wave.open(str(src)) as handle:
        rate = handle.getframerate() or SAMPLE_RATE
        channels = handle.getnchannels()
        writer = WavWriter(dst, rate, channels)
        try:
            while True:
                data = handle.readframes(rate * 30)
                if not data:
                    break
                block = numpy.frombuffer(data, dtype="<i2").astype(numpy.float32) * gain
                numpy.clip(block, -32_768, 32_767, out=block)
                writer.write(block.astype("<i2").tobytes())
        finally:
            writer.close()
    return gain


def format_minutes_left(seconds: float) -> str:
    if seconds < 60:
        return "不到 1 分钟"
    if seconds < 3_600:
        return f"约 {seconds / 60:.0f} 分钟"
    return f"约 {seconds / 3_600:.1f} 小时"


HALLUCINATION_PATTERNS = [
    r"请不吝(点赞|赐教)",
    r"打赏支持",
    r"明镜与点点栏目",
    r"字幕(由|组|組|志愿者).{0,12}(提供|制作|翻译)",
    r"amara\.org|字幕君",
    r"^(谢谢|感谢)(大家)?(观看|收看|收聽|聆听|收听)[。！!]?$",
    r"^(嗯|啊|呃|哦|噢|唔)+[。，,！!？?]?$",
]
PROMO_WORDS = ("点赞", "订阅", "转发", "打赏", "投币", "关注", "分享")
MAX_REPEATS = 2


def is_hallucination(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if sum(word in stripped for word in PROMO_WORDS) >= 2:
        return True
    return any(re.search(pattern, stripped) for pattern in HALLUCINATION_PATTERNS)


def diarization_available() -> bool:
    if not (SEGMENTATION_MODEL.exists() and EMBEDDING_MODEL.exists()):
        return False
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return True


def diarize(path: Path, progress=None) -> list[tuple[float, float, int]]:
    import numpy
    import sherpa_onnx

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEGMENTATION_MODEL)),
            num_threads=2,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMBEDDING_MODEL), num_threads=2),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=DIARIZE_THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("说话人分离模型配置无效")
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    with wave.open(str(path)) as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    if rate != engine.sample_rate:
        raise RuntimeError(f"说话人分离需要 {engine.sample_rate}Hz 音频，这个文件是 {rate}Hz")
    samples = numpy.frombuffer(raw, dtype="<i2").astype(numpy.float32) / 32_768.0
    del raw
    result = engine.process(samples, callback=progress) if progress else engine.process(samples)
    return [(item.start, item.end, item.speaker) for item in result.sort_by_start_time()]


def label_speakers(
    spans: list[tuple[float, float]], turns: list[tuple[float, float, int]]
) -> list[int | None]:
    labels: list[int | None] = []
    for start, end in spans:
        best: int | None = None
        best_overlap = 0.0
        for turn_start, turn_end, speaker in turns:
            overlap = min(end, turn_end) - max(start, turn_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best = speaker
        labels.append(best)
    return labels


def speaker_name(label: int | None) -> str:
    return f"说话人{label + 1}" if label is not None else "未知说话人"


def transcription_quality(scores: list[float]) -> str:
    if not scores:
        return "无法评估"
    average = sum(scores) / len(scores)
    if average < -0.6:
        return f"较低（平均置信分 {average:.2f}，纪要仅可作参考）"
    if average < -0.35:
        return f"一般（平均置信分 {average:.2f}，请核对专有名词和数字）"
    return f"较好（平均置信分 {average:.2f}）"


def fallback_minutes(transcript: str, meeting_title: str, audio_path: Path) -> str:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[。！？!?])", transcript)
        if item.strip()
    ]
    decisions = [s for s in sentences if re.search(r"决定|确定|结论|同意|通过", s)]
    actions = [s for s in sentences if re.search(r"负责|跟进|完成|截止|需要|请|行动", s)]
    questions = [s for s in sentences if re.search(r"待定|确认|问题|讨论|是否|不确定", s)]

    def bullets(items: list[str], limit: int = 12) -> str:
        return "\n".join(f"- {item}" for item in items[:limit]) or "- 暂未自动识别，请查看逐字稿补充。"

    overview = "".join(sentences[:8]) or "未识别到有效语音，请检查原始录音。"
    return (
        f"# {meeting_title}会议纪要\n\n"
        f"- 时间：{datetime.now():%Y-%m-%d %H:%M}\n"
        f"- 原始录音：`{audio_path.name}`\n"
        "- 生成方式：本地规则兜底（本地大模型未成功运行）\n\n"
        f"## 会议摘要\n\n{overview}\n\n"
        f"## 主要决定\n\n{bullets(decisions)}\n\n"
        f"## 行动项\n\n{bullets(actions)}\n\n"
        f"## 待确认问题\n\n{bullets(questions)}\n"
    )


def split_text(text: str, max_chars: int = 6_000) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > max_chars:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks or [text]


def ollama_models() -> set[str]:
    # This fixed endpoint is loopback-only; callers cannot choose a URL or scheme.
    with request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:  # nosec B310
        models = json.loads(response.read().decode("utf-8")).get("models", [])
    names = set()
    for item in models:
        name = item.get("name", "")
        names.add(name)
        names.add(name.removesuffix(":latest"))
    return names


def ollama_ready() -> tuple[bool, str]:
    try:
        names = ollama_models()
    except Exception as error:
        return False, f"本地 Ollama 没有在运行（{error}）"
    if OLLAMA_MODEL in names:
        return True, ""
    return False, f"Ollama 里没有 {OLLAMA_MODEL} 模型，请先执行 ollama pull {OLLAMA_MODEL}"


def pick_chunk_model() -> str:
    try:
        return OLLAMA_CHUNK_MODEL if OLLAMA_CHUNK_MODEL in ollama_models() else OLLAMA_MODEL
    except Exception:
        return OLLAMA_MODEL


def run_ollama(prompt: str, max_tokens: int = 900, model: str = "") -> str:
    payload = json.dumps(
        {
            "model": model or OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
    ).encode("utf-8")
    api_request = request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        # api_request always targets the fixed loopback-only URL above.
        with request.urlopen(api_request, timeout=600) as response:  # nosec B310
            result = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        raise RuntimeError(f"Ollama（{model or OLLAMA_MODEL}）调用失败：{error}") from error
    output = result.get("response", "").strip()
    if not output:
        raise RuntimeError(result.get("error") or "Ollama 没有返回内容")
    return output


def stop_ollama_model(model: str) -> None:
    executable = shutil.which("ollama")
    if executable:
        subprocess.run(
            [executable, "stop", model],
            capture_output=True,
            timeout=30,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def clean_markdown(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*\n?(.*?)\n?```", value, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else value


def generate_minutes(
    transcript: str,
    meeting_title: str,
    audio_path: Path,
    quality: str = "未提供",
    speakers: int = 0,
    progress=None,
) -> str:
    speaker_rule = (
        f"逐字稿里的“说话人1…说话人{speakers}”是机器自动分离的结果，可能分错或把两个人并成一个。"
        "可以用它区分谁说了什么，但绝不能推断他们的真实姓名、职务或所属部门；"
        "只有当发言人在原文里自报姓名或被别人当面称呼时，才可以写出姓名。\n"
        if speakers >= 2
        else ""
    )
    chunks = split_text(transcript)
    chunk_model = pick_chunk_model() if len(chunks) > 1 else OLLAMA_MODEL
    if len(chunks) == 1:
        combined = f"带时间戳的完整逐字稿：\n{transcript}"
    else:
        chunk_notes: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""你是一名严谨的中文会议秘书。下面是带时间戳的会议逐字稿第 {index}/{len(chunks)} 部分。
整份转写的质量评估为：{quality}。
{speaker_rule}只允许提取逐字稿明确表达的事实。不得猜测或补充人物身份、业务背景、金额含义、系统全称、决定、负责人或期限。
转写不通顺、含义不确定或疑似错字时，原样保留并放入“转写疑点”，不要替它编造合理解释。
每条决定和行动项必须附上时间戳及简短原文证据；没有明确证据就写“未识别到明确内容”。
请在 800 字以内输出：本段要点、明确决定、行动项、待确认问题、转写疑点。

逐字稿：
{chunk}
"""
            if progress is not None:
                progress(index, len(chunks) + 1)
            chunk_notes.append(run_ollama(prompt, max_tokens=1_200, model=chunk_model))
        combined = "\n\n".join(
            f"### 分段记录 {index}\n{note}"
            for index, note in enumerate(chunk_notes, start=1)
        )

    final_prompt = f"""你是一名严谨的中文会议秘书。请把下面的分段记录合并为最终会议纪要。
整份转写的质量评估为：{quality}。
{speaker_rule}必须使用 Markdown，并严格包含这些二级标题：会议摘要、讨论要点、主要决定、行动项、待确认问题、转写疑点。
行动项使用表格，列为“时间戳、负责人、任务、期限、原文证据”；原文没有的信息写“未明确”。
负责人一列：原文出现姓名就写姓名，只知道是谁说的就写“说话人N”，都没有就写“未明确”。
每条主要决定也必须包含时间戳和简短原文证据。没有明确决定或行动项时，明确写“未识别到明确内容”。
不得推断人物身份、业务背景、金额含义或系统全称。合并重复项，但不要补充逐字稿中不存在的事实。
总字数不超过 2000 字。
标题为“{meeting_title}会议纪要”。

分段记录：
{combined}
"""
    if chunk_model != OLLAMA_MODEL:
        stop_ollama_model(chunk_model)
    if progress is not None:
        progress(len(chunks) + 1, len(chunks) + 1)
    minutes = clean_markdown(run_ollama(final_prompt, max_tokens=3_000, model=OLLAMA_MODEL))
    metadata = (
        f"<!-- 本地生成；音频未上传 -->\n"
        f"- 会议时间：{datetime.now():%Y-%m-%d %H:%M}\n"
        f"- 原始录音：`{audio_path.name}`\n"
        f"- 完整逐字稿：`逐字稿.txt`\n\n"
        f"- 转写质量：{quality}\n\n"
    )
    return metadata + minutes + "\n"


class WavWriter:
    def __init__(self, path: Path, samplerate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.path = path
        self.samplerate = samplerate
        self.channels = channels
        self.frame_bytes = SAMPLE_WIDTH * channels
        self.frames = 0
        self._file = open(path, "wb")
        self._file.write(self._header(0))

    def _header(self, data_size: int) -> bytes:
        byte_rate = self.samplerate * self.frame_bytes
        return (
            b"RIFF"
            + struct.pack("<I", 36 + data_size)
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, self.channels, self.samplerate, byte_rate, self.frame_bytes, SAMPLE_WIDTH * 8)
            + b"data"
            + struct.pack("<I", data_size)
        )

    def write(self, data: bytes) -> None:
        if self.frames * self.frame_bytes + len(data) > MAX_WAV_DATA_BYTES:
            raise RuntimeError("录音文件已达到 WAV 格式的 4GB 上限，请分段录制")
        self._file.write(data)
        self.frames += len(data) // self.frame_bytes

    def flush(self) -> None:
        data_size = self.frames * self.frame_bytes
        end = self._file.tell()
        self._file.seek(0)
        self._file.write(self._header(data_size))
        self._file.seek(end)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        if self._file.closed:
            return
        try:
            self.flush()
        finally:
            self._file.close()


class Recorder:
    def __init__(
        self,
        path: Path | None,
        device: int | None = None,
        extra_settings=None,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        keep_preroll: bool = False,
        preroll: bytes = b"",
        preroll_ended_at: float = 0.0,
        label: str = "主录音",
    ):
        self.path = path
        self.device = device
        self.extra_settings = extra_settings
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = max(1, samplerate // 10)
        self.label = label
        self.preroll_seconds = 0.0
        self.preroll_ended_at = preroll_ended_at
        self.audio_started_at = 0.0
        self.live_started_at = 0.0
        self._initial = preroll
        self._preroll: collections.deque[bytes] | None = (
            collections.deque(maxlen=int(PREROLL_SEC * samplerate / self.blocksize)) if keep_preroll else None
        )
        self.frames_written = 0
        self.level_peak = 0
        self.max_peak = 0
        self.overflowed = False
        self.restarts = 0
        self.capture_error: str | None = None
        self.write_error: str | None = None
        self.stalled_seconds = 0.0
        self.events: list[str] = []
        self.last_data_at = 0.0
        self.last_capture_ended_at = 0.0
        self.started_at = 0.0
        self.stopped_at = 0.0
        self._writer: WavWriter | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._stream: sd.RawInputStream | None = None
        self._consumer: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None
        self._stopping = threading.Event()
        self._lock = threading.RLock()

    @property
    def error(self) -> str | None:
        # A recovered capture stall must never hide a file that stopped being written.
        if self.stalled_seconds > 0:
            gap_error = f"录音采集曾中断，时间轴丢失约 {self.stalled_seconds:.1f} 秒"
        else:
            gap_error = None
        return self.write_error or self.capture_error or gap_error

    @property
    def recorded_seconds(self) -> float:
        return self.frames_written / self.samplerate

    @property
    def expected_seconds(self) -> float:
        if not self.started_at:
            return self.recorded_seconds
        end = self.stopped_at or time.monotonic()
        return self.preroll_seconds + max(0.0, end - self.started_at)

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self.last_data_at if self.last_data_at else 0.0

    def start(self) -> None:
        if self.path is not None:
            # The file is always mono; stereo-only devices are downmixed in the callback.
            self._writer = WavWriter(self.path, self.samplerate, 1)
            if self._initial:
                self._writer.write(self._initial)
                self._writer.flush()
                self.frames_written += len(self._initial) // SAMPLE_WIDTH
                self.preroll_seconds = self.frames_written / self.samplerate
                if self.preroll_ended_at:
                    self.audio_started_at = self.preroll_ended_at - self.preroll_seconds
        self.started_at = time.monotonic()
        self.last_data_at = self.started_at
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()
        try:
            self._open_stream()
        except Exception:
            self._stopping.set()
            self._queue.put(None)
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            raise
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def _open_stream(self) -> None:
        stream = sd.RawInputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="int16",
            blocksize=self.blocksize,
            device=self.device,
            extra_settings=self.extra_settings,
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def _callback(self, indata, frames, time_info, status) -> None:
        if status.input_overflow:
            self.overflowed = True
        captured_at = time.monotonic()
        self.last_data_at = captured_at
        block_seconds = frames / self.samplerate
        try:
            adc_time = float(time_info.inputBufferAdcTime)
            callback_time = float(time_info.currentTime)
            if not math.isfinite(adc_time) or not math.isfinite(callback_time):
                raise ValueError("invalid PortAudio capture timestamp")
            block_started_at = captured_at + adc_time - callback_time
        except (AttributeError, TypeError, ValueError):
            block_started_at = captured_at - block_seconds
        self.last_capture_ended_at = block_started_at + block_seconds
        if not self.live_started_at:
            self.live_started_at = block_started_at
            if self.preroll_ended_at and self.preroll_seconds:
                self.audio_started_at = self.preroll_ended_at - self.preroll_seconds
                gap_frames = round(max(0.0, self.live_started_at - self.preroll_ended_at) * self.samplerate)
                if gap_frames:
                    self._queue.put(bytes(gap_frames * SAMPLE_WIDTH))
            else:
                self.audio_started_at = self.live_started_at
        self._queue.put(downmix_to_mono(bytes(indata), self.channels))

    def _consume(self) -> None:
        try:
            self._consume_loop()
        finally:
            # The consumer owns the writer: finalize the header even if stop() gave up waiting.
            writer = self._writer
            self._writer = None
            if writer is not None:
                try:
                    writer.close()
                except Exception as error:
                    self._note(f"关闭录音文件失败：{error}")
                    self.write_error = f"关闭录音文件失败：{error}"

    def _consume_loop(self) -> None:
        next_flush = time.monotonic() + FLUSH_INTERVAL_SEC
        while True:
            try:
                data = self._queue.get(timeout=0.5)
            except queue.Empty:
                data = b""
            if data is None:
                break
            if data:
                if self._preroll is not None:
                    self._preroll.append(data)
                peak = block_peak(data)
                self.level_peak = peak
                self.max_peak = max(self.max_peak, peak)
                if self.path is not None:
                    if self._writer is None:
                        continue
                    try:
                        self._writer.write(data)
                    except Exception as error:
                        self._note(f"写入录音文件失败：{error}")
                        self.write_error = f"写入录音文件失败：{error}"
                        failed_writer = self._writer
                        self._writer = None
                        try:
                            failed_writer.close()
                        except Exception:
                            pass
                        continue
                self.frames_written += len(data) // SAMPLE_WIDTH
            if self._writer is not None and time.monotonic() >= next_flush:
                try:
                    self._writer.flush()
                except Exception as error:
                    self._note(f"落盘失败：{error}")
                    self.write_error = f"录音落盘失败：{error}"
                    failed_writer = self._writer
                    self._writer = None
                    try:
                        failed_writer.close()
                    except Exception:
                        pass
                next_flush = time.monotonic() + FLUSH_INTERVAL_SEC

    def _watch(self) -> None:
        while not self._stopping.wait(0.5):
            if self._stream is not None and self.idle_seconds > STALL_RESTART_SEC:
                self._restart_stream()

    def _restart_stream(self) -> None:
        with self._lock:
            if self._stopping.is_set():
                return
            stamp = timestamp(self.recorded_seconds)
            if self._stream is not None:
                try:
                    self._stream.abort(ignore_errors=True)
                    self._stream.close(ignore_errors=True)
                except Exception:
                    pass
                self._stream = None
            now = time.monotonic()
            self.stalled_seconds += now - self.last_data_at
            self.last_data_at = now
            try:
                self._open_stream()
                self.restarts += 1
                self._note(f"[{stamp}] 麦克风采集中断，已自动重开（第 {self.restarts} 次）")
                self.capture_error = None
            except Exception as error:
                self._note(f"[{stamp}] 麦克风采集中断，重开失败：{error}")
                self.capture_error = f"麦克风采集中断且暂时无法恢复：{error}"

    def preroll_bytes(self) -> bytes:
        if self._preroll is None:
            return b""
        return b"".join(list(self._preroll))

    def _note(self, message: str) -> None:
        self.events.append(message)

    def stop(self) -> None:
        self.stopped_at = time.monotonic()
        self._stopping.set()
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
        if self._watchdog is not None:
            self._watchdog.join(timeout=3)
            self._watchdog = None
        self._queue.put(None)
        if self._consumer is not None:
            self._consumer.join(timeout=15)
            if self._consumer.is_alive():
                self.write_error = "音频写盘线程未能在 15 秒内完成，录音文件可能缺少末尾内容"
                self._note(self.write_error)
                raise RuntimeError(self.write_error)
            self._consumer = None
        if self._writer is not None:
            try:
                self._writer.close()
            finally:
                self._writer = None


def input_devices() -> list[tuple[int, str]]:
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0 and not is_system_audio_input(device["name"]):
            api = sd.query_hostapis(device["hostapi"])["name"]
            devices.append((index, f"{device['name']}  [{api}]"))
    return devices


def preferred_microphone_device(default_index: int, available_indices: list[int]) -> int | None:
    if not available_indices:
        return None
    try:
        devices = sd.query_devices()
        default_name = devices[default_index]["name"].strip() if 0 <= default_index < len(devices) else ""
        matching = [
            index
            for index in available_indices
            if default_name and devices[index]["name"].strip() == default_name
        ]
        if matching:
            return min(
                matching,
                key=lambda index: (
                    sd.query_hostapis(devices[index]["hostapi"])["name"] != "Windows WASAPI",
                    index != default_index,
                ),
            )
    except Exception:
        pass
    return default_index if default_index in available_indices else available_indices[0]


SYSTEM_AUDIO_UNRECOGNISED_SUFFIX = "（非系统回放端点，仅虚拟声卡输出可用）"


def system_input_devices() -> list[tuple[int, str]]:
    devices = []
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] <= 0:
            continue
        api = sd.query_hostapis(device["hostapi"])["name"]
        name = f"{device['name']}  [{api}]"
        if not is_system_audio_input(device["name"]):
            # Virtual cables (VB-Cable, VoiceMeeter) carry system audio under arbitrary names.
            name += SYSTEM_AUDIO_UNRECOGNISED_SUFFIX
        devices.append((index, name))
    devices.sort(key=lambda item: (not is_system_audio_input(item[1]), *system_audio_priority(item[1]), item[0]))
    return devices


def input_capture_settings(device_index: int) -> tuple[int, int, object | None]:
    device = sd.query_devices(device_index)
    api_name = sd.query_hostapis(device["hostapi"])["name"]
    extra: object | None = sd.WasapiSettings(auto_convert=True) if api_name == "Windows WASAPI" else None

    def supports(samplerate: int, channels: int) -> bool:
        try:
            sd.check_input_settings(
                device=device_index,
                samplerate=samplerate,
                channels=channels,
                dtype="int16",
                extra_settings=extra,
            )
            return True
        except Exception:
            return False

    samplerate, channels = select_input_format(
        device["default_samplerate"], device["max_input_channels"], supports
    )
    return samplerate, channels, extra


def meeting_audio_paths(meeting_dir: Path, online: bool) -> dict[str, Path | None]:
    if online:
        return {
            "primary": meeting_dir / "麦克风录音.wav",
            "backup": meeting_dir / "麦克风备份.wav",
            "system": meeting_dir / "系统声音.wav",
            "complete": meeting_dir / "完整录音.wav",
        }
    primary = meeting_dir / "原始录音.wav"
    return {
        "primary": primary,
        "backup": meeting_dir / "备份录音.wav",
        "system": None,
        "complete": primary,
    }


def choose_microphone_recording(primary: Recorder, backup: Recorder | None) -> Path:
    if primary.path is None:
        raise ValueError("主录音没有文件路径")
    if backup is None or backup.path is None:
        return primary.path

    def score(recorder: Recorder) -> tuple[bool, bool, float]:
        return recorder.max_peak >= SILENCE_PEAK, recorder.error is None, recorder.recorded_seconds

    return backup.path if score(backup) > score(primary) else primary.path


def track_start_offsets(*recorders: Recorder) -> list[float]:
    """Delay tracks so their first captured samples share one monotonic timeline."""
    origins = [recorder.audio_started_at for recorder in recorders]
    if any(origin <= 0 for origin in origins):
        raise RuntimeError("录音轨缺少实际采集起点，无法可靠对齐")
    base = min(origins)
    return [origin - base for origin in origins]


def recording_failures(*recorders: Recorder | None) -> list[str]:
    return [f"{recorder.label}：{recorder.error}" for recorder in recorders if recorder and recorder.error]


def stop_recorders_safely(*recorders: Recorder | None) -> list[str]:
    errors = []
    for recorder in recorders:
        if recorder is None:
            continue
        try:
            recorder.stop()
        except Exception as error:
            if recorder.error is None:
                recorder.write_error = str(error)
            errors.append(f"{recorder.label}：{error}")
    return errors


def wasapi_index() -> int | None:
    for index in range(len(sd.query_hostapis())):
        if sd.query_hostapis(index)["name"] == "Windows WASAPI":
            return index
    return None


def backup_candidates(primary: int | None) -> list[tuple[int, object | None]]:
    try:
        devices = sd.query_devices()
        primary_index = sd.default.device[0] if primary is None else primary
        primary_name = devices[primary_index]["name"].strip()
        primary_api = devices[primary_index]["hostapi"]
    except Exception:
        return []
    wasapi = wasapi_index()
    ranked: list[tuple[int, int]] = []
    for index, device in enumerate(devices):
        if (
            device["max_input_channels"] <= 0
            or index == primary_index
            or is_system_audio_input(device["name"])
        ):
            continue
        same_device = device["name"].strip() == primary_name
        if same_device and device["hostapi"] != primary_api:
            ranked.append((0 if device["hostapi"] == wasapi else 1, index))
        elif not same_device:
            ranked.append((2 if device["hostapi"] == wasapi else 3, index))
    ranked.sort()
    result = []
    for _rank, index in ranked[:4]:
        extra = sd.WasapiSettings(auto_convert=True) if devices[index]["hostapi"] == wasapi else None
        result.append((index, extra))
    return result


class MeetingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("会议录音与纪要（高质量本地版）")
        self.geometry("720x650")
        self.minsize(700, 630)
        self.recorder: Recorder | None = None
        self.backup: Recorder | None = None
        self.monitor: Recorder | None = None
        self.system_recorder: Recorder | None = None
        self.system_monitor: Recorder | None = None
        self.meeting_dir: Path | None = None
        self.audio_path: Path | None = None
        self.meeting_title = "会议"
        self.processing = False
        self.devices: list[tuple[int, str]] = []
        self.system_devices: list[tuple[int, str]] = []
        self.meeting_mode = tk.StringVar(value="offline")
        self._cancel = threading.Event()
        self._transcribe_started = 0.0
        self.stage_count = 2
        self.diarize_now = False
        self.diarize_enabled = tk.BooleanVar(value=diarization_available())
        self._silence_since: float | None = None
        self._system_silence_since: float | None = None
        self._silence_warned = False
        self._sleep_asserted_at = 0.0
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()
        self.after(200, self._reload_devices)
        self.after(300, self._tick)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="会议录音与纪要", font=("Microsoft YaHei UI", 20, "bold")).pack(anchor="w")
        ttk.Label(frame, text="音频和文字仅保存在本机；线上模式会同时保留麦克风、系统声音和完整混合录音。", foreground="#555").pack(anchor="w", pady=(4, 14))

        title_row = ttk.Frame(frame)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="会议名称：").pack(side="left")
        self.title_entry = ttk.Entry(title_row)
        self.title_entry.insert(0, "会议")
        self.title_entry.pack(side="left", fill="x", expand=True)

        mode_row = ttk.Frame(frame)
        mode_row.pack(fill="x", pady=(10, 0))
        ttk.Label(mode_row, text="会议类型：").pack(side="left")
        self.offline_mode = ttk.Radiobutton(
            mode_row,
            text="线下会议（麦克风）",
            value="offline",
            variable=self.meeting_mode,
            command=self._mode_changed,
        )
        self.offline_mode.pack(side="left", padx=(0, 14))
        self.online_mode = ttk.Radiobutton(
            mode_row,
            text="线上会议（麦克风 + 电脑声音）",
            value="online",
            variable=self.meeting_mode,
            command=self._mode_changed,
        )
        self.online_mode.pack(side="left")

        device_row = ttk.Frame(frame)
        device_row.pack(fill="x", pady=(10, 0))
        ttk.Label(device_row, text="本人/现场麦克风：").pack(side="left")
        self.device_box = ttk.Combobox(device_row, state="readonly", values=[])
        self.device_box.pack(side="left", fill="x", expand=True)
        self.device_box.bind("<<ComboboxSelected>>", lambda _event: self._restart_monitor())
        ttk.Button(device_row, text="刷新", width=6, command=self._reload_devices).pack(side="left", padx=(6, 0))

        system_row = ttk.Frame(frame)
        system_row.pack(fill="x", pady=(8, 0))
        ttk.Label(system_row, text="电脑系统声音：").pack(side="left")
        self.system_box = ttk.Combobox(system_row, state="disabled", values=[])
        self.system_box.pack(side="left", fill="x", expand=True)
        self.system_box.bind("<<ComboboxSelected>>", lambda _event: self._restart_monitor())

        level_row = ttk.Frame(frame)
        level_row.pack(fill="x", pady=(12, 0))
        ttk.Label(level_row, text="麦克风电平：").pack(side="left")
        self.level_bar = ttk.Progressbar(level_row, mode="determinate", maximum=100)
        self.level_bar.pack(side="left", fill="x", expand=True)
        self.level_text = ttk.Label(level_row, text="-- dBFS", width=12)
        self.level_text.pack(side="left", padx=(8, 0))

        system_level_row = ttk.Frame(frame)
        system_level_row.pack(fill="x", pady=(6, 0))
        ttk.Label(system_level_row, text="系统声电平：").pack(side="left")
        self.system_level_bar = ttk.Progressbar(system_level_row, mode="determinate", maximum=100)
        self.system_level_bar.pack(side="left", fill="x", expand=True)
        self.system_level_text = ttk.Label(system_level_row, text="线下模式", width=12)
        self.system_level_text.pack(side="left", padx=(8, 0))

        self.level_hint = ttk.Label(frame, text="正在检查麦克风……", foreground="#555", wraplength=600)
        self.level_hint.pack(anchor="w", pady=(6, 0))

        self.diarize_check = ttk.Checkbutton(
            frame,
            text="识别谁在说话（会后额外花约录音时长的 1/10）",
            variable=self.diarize_enabled,
        )
        self.diarize_check.pack(anchor="w", pady=(6, 0))
        if not diarization_available():
            self.diarize_check.config(state="disabled", text="识别谁在说话（缺少 models/ 里的说话人模型，已禁用）")

        self.timer_label = ttk.Label(frame, text="00:00:00", font=("Consolas", 26, "bold"))
        self.timer_label.pack(anchor="center", pady=10)

        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=4)
        self.start_button = ttk.Button(buttons, text="开始录音", command=self.start_recording)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.stop_button = ttk.Button(buttons, text="结束并生成纪要", command=self.stop_recording, state="disabled")
        self.stop_button.pack(side="left", fill="x", expand=True, padx=(6, 0))

        extra = ttk.Frame(frame)
        extra.pack(fill="x", pady=(8, 10))
        self.open_button = ttk.Button(extra, text="打开结果文件夹", command=self.open_results)
        self.open_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.existing_button = ttk.Button(extra, text="处理已有录音…", command=self.open_existing)
        self.existing_button.pack(side="left", fill="x", expand=True, padx=(6, 0))
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status_label = ttk.Label(
            frame,
            text=f"准备就绪。转写：{WHISPER_MODEL}；纪要：{OLLAMA_MODEL}。请先确认所需电平条会动，再开始正式录音。",
            wraplength=600,
        )
        self.status_label.pack(anchor="w", pady=(9, 0))

    def _selected_device(self) -> int | None:
        index = self.device_box.current()
        if 0 <= index < len(self.devices):
            return self.devices[index][0]
        return None

    def _selected_system_device(self) -> int | None:
        index = self.system_box.current()
        if 0 <= index < len(self.system_devices):
            return self.system_devices[index][0]
        return None

    def _is_online(self) -> bool:
        return self.meeting_mode.get() == "online"

    def _mode_changed(self) -> None:
        self.system_box.config(state="readonly" if self._is_online() and self.system_devices else "disabled")
        self.system_level_text.config(text="静音" if self._is_online() else "线下模式")
        self._restart_monitor()

    def _reload_devices(self) -> None:
        self._stop_monitor()
        if self.recorder is None:
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass
        try:
            self.devices = input_devices()
            self.system_devices = system_input_devices()
            default = sd.default.device[0]
        except Exception as error:
            self.devices = []
            self.system_devices = []
            self.device_box.config(values=[])
            self.system_box.config(values=[], state="disabled")
            self.level_hint.config(text=f"麦克风不可用：{error}", foreground="#c00000")
            self.start_button.config(state="disabled")
            return
        self.device_box.config(values=[name for _index, name in self.devices])
        self.system_box.config(values=[name for _index, name in self.system_devices])
        if not self.devices:
            self.level_hint.config(text="没有找到任何录音设备。", foreground="#c00000")
            self.start_button.config(state="disabled")
            return
        preferred = preferred_microphone_device(default, [index for index, _name in self.devices])
        chosen = next(
            (position for position, (index, _name) in enumerate(self.devices) if index == preferred),
            0,
        )
        self.device_box.current(chosen)
        if self.system_devices:
            self.system_box.current(0)
        self.system_box.config(state="readonly" if self._is_online() and self.system_devices else "disabled")
        if self.recorder is None:
            self._restart_monitor()

    def _restart_monitor(self) -> None:
        self._stop_monitor()
        if self.recorder is not None:
            return
        device = self._selected_device()
        if device is None:
            self.start_button.config(state="disabled")
            return
        try:
            rate, channels, extra = input_capture_settings(device)
            monitor = Recorder(
                None,
                device=device,
                extra_settings=extra,
                samplerate=rate,
                channels=channels,
                keep_preroll=True,
                label="麦克风试音",
            )
            monitor.start()
        except Exception as error:
            self.level_hint.config(text=f"无法打开麦克风：{error}", foreground="#c00000")
            self.start_button.config(state="disabled")
            return
        self.monitor = monitor
        if self._is_online():
            system_device = self._selected_system_device()
            if system_device is None:
                monitor.stop()
                self.monitor = None
                self.start_button.config(state="disabled")
                self.level_hint.config(
                    text="线上模式没有找到任何可用的系统声音输入。请在 Windows 录音设备中启用“立体声混音”，再点刷新。",
                    foreground="#c00000",
                )
                return
            try:
                rate, channels, extra = input_capture_settings(system_device)
                system_monitor = Recorder(
                    None,
                    device=system_device,
                    extra_settings=extra,
                    samplerate=rate,
                    channels=channels,
                    keep_preroll=True,
                    label="系统声音试音",
                )
                system_monitor.start()
            except Exception as error:
                monitor.stop()
                self.monitor = None
                self.start_button.config(state="disabled")
                self.level_hint.config(text=f"无法打开电脑系统声音：{error}", foreground="#c00000")
                return
            self.system_monitor = system_monitor
            self.level_hint.config(
                text="线上试音中：请自己说话并播放一段电脑声音，确认两条电平都会动。",
                foreground="#555",
            )
        else:
            self.level_hint.config(text="线下试音中：请对着麦克风说话，确认麦克风电平明显跳动。", foreground="#555")
        self.start_button.config(state="normal")

    def _stop_monitor(self) -> None:
        if self.monitor is not None:
            try:
                self.monitor.stop()
            except Exception:
                pass
            self.monitor = None
        if self.system_monitor is not None:
            try:
                self.system_monitor.stop()
            except Exception:
                pass
            self.system_monitor = None

    @staticmethod
    def _show_level(source: Recorder | None, bar, label, idle_text: str = "—") -> None:
        if source is None:
            bar.config(value=0)
            label.config(text=idle_text)
            return
        peak = 0 if source.idle_seconds > STALL_WARN_SEC else source.level_peak
        bar.config(value=level_percent(peak))
        label.config(text=f"{peak_to_dbfs(peak):.0f} dBFS" if peak >= SILENCE_PEAK else "静音")

    def _tick(self) -> None:
        source = self.recorder or self.monitor
        self._show_level(source, self.level_bar, self.level_text)
        system_source = self.system_recorder or self.system_monitor
        self._show_level(
            system_source,
            self.system_level_bar,
            self.system_level_text,
            "静音" if self._is_online() else "线下模式",
        )
        if self.recorder is not None:
            self._update_recording_ui(self.recorder)
        elif self.monitor is not None:
            self._update_monitor_ui(self.monitor)
        self.after(200, self._tick)

    def _update_monitor_ui(self, monitor: Recorder) -> None:
        if monitor.idle_seconds > STALL_WARN_SEC:
            self.level_hint.config(text="麦克风没有在送数据，请点“刷新”或换一个设备。", foreground="#c00000")
        elif monitor.max_peak < SILENCE_PEAK:
            self.level_hint.config(
                text="还没有听到任何声音。请说句话试试；若电平条始终不动，检查 Windows 声音设置里麦克风是否被静音或音量为 0。",
                foreground="#b06000",
            )
        else:
            system = self.system_monitor
            if self._is_online() and (system is None or system.max_peak < SILENCE_PEAK):
                self.level_hint.config(
                    text="麦克风正常；还没听到电脑声音。请播放一段视频或让线上参会者说话，确认系统声电平会动。",
                    foreground="#b06000",
                )
            else:
                self.level_hint.config(text="所需音频输入工作正常，可以开始录音。", foreground="#207020")

    def _backup_note(self) -> str:
        backup = self.backup
        if backup is None:
            return "；无备份路"
        if backup.max_peak >= SILENCE_PEAK:
            return "；备份路有声音"
        return "；备份路也是静音"

    def _update_recording_ui(self, recorder: Recorder) -> None:
        now = time.monotonic()
        if now - self._sleep_asserted_at > SLEEP_REASSERT_SEC:
            prevent_system_sleep(True)
            self._sleep_asserted_at = now

        self.timer_label.config(text=timestamp(recorder.recorded_seconds))
        drift = (now - recorder.started_at) - recorder.recorded_seconds

        if recorder.level_peak >= SILENCE_PEAK:
            self._silence_since = None
            self._silence_warned = False
        elif self._silence_since is None:
            self._silence_since = now

        system = self.system_recorder
        if system is not None and system.level_peak >= SILENCE_PEAK:
            self._system_silence_since = None
        elif system is not None and self._system_silence_since is None:
            self._system_silence_since = now

        if recorder.error:
            self.level_hint.config(text=f"⚠ {recorder.error}", foreground="#c00000")
        elif system is not None and system.error:
            self.level_hint.config(text=f"⚠ 系统声音：{system.error}", foreground="#c00000")
        elif recorder.idle_seconds > STALL_WARN_SEC:
            self.level_hint.config(
                text=f"⚠ 麦克风已 {recorder.idle_seconds:.0f} 秒没有送来数据，正在自动重连（已重连 {recorder.restarts} 次）。",
                foreground="#c00000",
            )
        elif system is not None and system.idle_seconds > STALL_WARN_SEC:
            self.level_hint.config(
                text=f"⚠ 电脑系统声音已 {system.idle_seconds:.0f} 秒没有送来数据，正在自动重连。",
                foreground="#c00000",
            )
        elif (
            system is not None
            and self._system_silence_since is not None
            and now - self._system_silence_since > SILENCE_WARN_SEC
        ):
            self.level_hint.config(
                text="⚠ 电脑系统声音持续静音；请确认线上会议正在使用所选扬声器，并让远端播放测试声音。",
                foreground="#c00000",
            )
        elif self._silence_since is not None and now - self._silence_since > SILENCE_WARN_SEC:
            quiet = now - self._silence_since
            self.level_hint.config(
                text=f"⚠ 已连续 {quiet:.0f} 秒没有采到声音（录到的是静音）"
                f"{self._backup_note()}。请确认麦克风未被静音、音量不为 0，并让说话人离电脑近一些。",
                foreground="#c00000",
            )
            if not self._silence_warned and quiet > SILENCE_WARN_SEC * 2:
                self._silence_warned = True
                self.bell()
        elif drift > STALL_WARN_SEC:
            self.level_hint.config(text=f"⚠ 录音数据落后墙上时间 {drift:.0f} 秒，中间有丢失。", foreground="#c00000")
        else:
            note = f"（已自动重连 {recorder.restarts} 次）" if recorder.restarts else ""
            self.level_hint.config(
                text=f"正在录音，声音正常{note}{self._backup_note()}。", foreground="#207020"
            )

    def _take_preroll(self, system: bool = False) -> tuple[bytes, float]:
        monitor = self.system_monitor if system else self.monitor
        if system:
            self.system_monitor = None
        else:
            self.monitor = None
        if monitor is None:
            return b"", 0.0
        try:
            monitor.stop()
            data = monitor.preroll_bytes()
            return data, monitor.last_capture_ended_at if data else 0.0
        except Exception:
            return b"", 0.0

    def _start_backup(self, primary: int | None, backup_path: Path) -> Recorder | None:
        for index, _candidate_extra in backup_candidates(primary):
            try:
                rate, channels, extra = input_capture_settings(index)
                recorder = Recorder(
                    backup_path,
                    device=index,
                    extra_settings=extra,
                    samplerate=rate,
                    channels=channels,
                    label="麦克风备份",
                )
                recorder.start()
                return recorder
            except Exception:
                continue
        try:
            backup_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    def start_recording(self) -> None:
        recorder: Recorder | None = None
        system_recorder: Recorder | None = None
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(OUTPUT_DIR).free
            online = self._is_online()
            minimum_free = 5 * 1024**3 if online else MIN_FREE_BYTES
            usage_note = "线上三份音轨可能超过 1GB/小时" if online else "两路合计约 230MB/小时"
            if free < minimum_free and not messagebox.askyesno(
                "磁盘空间不多了",
                f"这个盘只剩 {free / 1024 ** 3:.1f} GB；{usage_note}。仍然开始吗？",
            ):
                return
            self.meeting_title = safe_name(self.title_entry.get())
            folder = f"{datetime.now():%Y%m%d-%H%M%S}-{self.meeting_title}"
            self.meeting_dir = OUTPUT_DIR / folder
            self.meeting_dir.mkdir(exist_ok=True)
            paths = meeting_audio_paths(self.meeting_dir, online)
            primary_path = paths["primary"]
            complete_path = paths["complete"]
            backup_path = paths["backup"]
            if primary_path is None or complete_path is None or backup_path is None:
                raise RuntimeError("无法创建录音文件路径")
            self.audio_path = complete_path
            device = self._selected_device()
            if device is None:
                raise RuntimeError("没有选择麦克风")
            preroll, preroll_ended_at = self._take_preroll()
            rate, channels, extra = input_capture_settings(device)
            recorder = Recorder(
                primary_path,
                device=device,
                extra_settings=extra,
                samplerate=rate,
                channels=channels,
                preroll=preroll,
                preroll_ended_at=preroll_ended_at,
                label="麦克风主录音",
            )
            recorder.start()
            if online:
                system_device = self._selected_system_device()
                system_path = paths["system"]
                if system_device is None or system_path is None:
                    raise RuntimeError("没有可用的电脑系统声音输入")
                system_preroll, system_preroll_ended_at = self._take_preroll(system=True)
                rate, channels, extra = input_capture_settings(system_device)
                system_recorder = Recorder(
                    system_path,
                    device=system_device,
                    extra_settings=extra,
                    samplerate=rate,
                    channels=channels,
                    preroll=system_preroll,
                    preroll_ended_at=system_preroll_ended_at,
                    label="系统声音",
                )
                system_recorder.start()
        except Exception as error:
            if recorder is not None:
                try:
                    recorder.stop()
                except Exception:
                    pass
            if system_recorder is not None:
                try:
                    system_recorder.stop()
                except Exception:
                    pass
            self.recorder = None
            self.system_recorder = None
            messagebox.showerror("无法开始录音", f"请确认麦克风和系统声音设备可用。\n\n{error}")
            self._restart_monitor()
            return

        self.recorder = recorder
        self.system_recorder = system_recorder
        self.backup = self._start_backup(device, backup_path)
        self._silence_since = None
        self._system_silence_since = None
        self._silence_warned = False
        self._sleep_asserted_at = time.monotonic()
        if not prevent_system_sleep(True):
            self.status_label.config(text="注意：系统拒绝了阻止睡眠的请求，请手动把电源计划设为“从不睡眠”。")
        else:
            note = "麦克风和系统声音正在分别保存" if online else "麦克风正在双路保存"
            self.status_label.config(text=f"正在录音：{note}。结果目录：{self.meeting_dir}")
        self.start_button.config(state="disabled")
        self.existing_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.title_entry.config(state="disabled")
        self.device_box.config(state="disabled")
        self.system_box.config(state="disabled")
        self.offline_mode.config(state="disabled")
        self.online_mode.config(state="disabled")

    def stop_recording(self) -> None:
        if self.recorder is None or self.audio_path is None or self.meeting_dir is None:
            return
        recorder = self.recorder
        backup = self.backup
        system = self.system_recorder
        self.recorder = None
        self.backup = None
        self.system_recorder = None
        self.stop_button.config(state="disabled")
        try:
            stop_recorders_safely(recorder, backup, system)
        finally:
            prevent_system_sleep(False)

        self._write_recording_log(recorder, backup, system)
        failures = recording_failures(recorder, backup, system)
        if failures:
            messagebox.showwarning(
                "录音写入异常",
                "以下音轨没有完整写入，请优先检查其他原轨和录音日志：\n\n" + "\n".join(failures),
            )
            microphone_failed = recorder.error is not None and (backup is None or backup.error is not None)
            system_failed = system is not None and system.error is not None
            if microphone_failed or system_failed:
                self.status_label.config(text="必要音轨写入失败，已停止自动处理；现有文件仍保留在结果目录。")
                self._restore_ready_controls()
                return
        invalid: list[tuple[Recorder, str]] = []
        for track in (recorder, backup, system):
            if track is None or track.path is None:
                continue
            try:
                verify_wav_file(track.path, track.expected_seconds)
            except Exception as error:
                invalid.append((track, f"{track.label}：{error}"))
        chosen_mic_path = choose_microphone_recording(recorder, backup)
        chosen_recorder = backup if backup is not None and chosen_mic_path == backup.path else recorder
        # A short system track would be mixed against the full microphone track and shift
        # every later remote sentence earlier, so it must block processing like the mic does.
        if any(track is chosen_recorder or track is system for track, _message in invalid):
            messagebox.showerror(
                "录音不可用",
                "必要音轨没有通过完整性校验，已停止自动处理。全部文件仍保留在结果目录，"
                "可用“处理已有录音…”手动重试。\n\n" + "\n".join(message for _track, message in invalid),
            )
            self.status_label.config(text="必要音轨校验失败，已停止自动处理；现有文件仍保留在结果目录。")
            self._restore_ready_controls()
            return
        if invalid:
            messagebox.showwarning(
                "录音文件校验失败",
                "以下音轨可能不完整，请保留全部文件并查看录音日志：\n\n"
                + "\n".join(message for _track, message in invalid),
            )
        if chosen_mic_path != recorder.path:
            messagebox.showinfo(
                "改用备份录音",
                "麦克风备份比主录音更完整，将用备份轨生成完整录音；两份原轨都会保留。",
            )
        mic_peak = chosen_recorder.max_peak
        chosen_path = chosen_mic_path
        if system is not None:
            self.status_label.config(text="原始音轨已保存，正在生成完整录音……")
            self.update_idletasks()
            try:
                self._mix_complete_recording(chosen_recorder, chosen_mic_path, system)
                chosen_path = self.audio_path
            except Exception as error:
                messagebox.showerror(
                    "无法生成完整录音",
                    f"麦克风和系统声音原轨仍已保留，但完整录音生成失败。\n\n{error}",
                )
                self._restore_ready_controls()
                return

        system_peak = system.max_peak if system is not None else SILENCE_PEAK
        if mic_peak < SILENCE_PEAK or system_peak < SILENCE_PEAK:
            self.status_label.config(text="有音频来源几乎无声，请先试听原始音轨。")
            missing = []
            if mic_peak < SILENCE_PEAK:
                missing.append("麦克风")
            if system is not None and system_peak < SILENCE_PEAK:
                missing.append("电脑系统声音")
            if not messagebox.askyesno(
                "录音几乎没有声音",
                f"{'、'.join(missing)}几乎没有声音。原始音轨已保留，请先试听检查。\n\n仍然要继续转写吗？",
            ):
                self._restore_ready_controls()
                return

        warning = "（录音期间检测到输入溢出，请重点检查音频）" if recorder.overflowed else ""
        self.status_label.config(text=f"录音已保存{warning}。正在准备转写……")
        self._start_processing(self.meeting_dir, chosen_path, self.meeting_title)

    def _mix_complete_recording(self, microphone: Recorder, microphone_path: Path, system: Recorder) -> None:
        if self.audio_path is None or system.path is None:
            raise RuntimeError("系统声音没有文件路径")
        failures = recording_failures(microphone, system)
        if failures:
            raise RuntimeError("必要音轨的时间轴不完整，不能可靠合成：\n" + "\n".join(failures))
        offsets = track_start_offsets(microphone, system)
        mix_wav_tracks([microphone_path, system.path], self.audio_path, start_offsets=offsets)
        expected = max(
            offsets[0] + microphone.recorded_seconds,
            offsets[1] + system.recorded_seconds,
        )
        verify_wav_file(self.audio_path, expected)

    def _restore_ready_controls(self) -> None:
        self.start_button.config(state="normal")
        self.existing_button.config(state="normal")
        self.title_entry.config(state="normal")
        self.device_box.config(state="readonly")
        self.system_box.config(state="readonly" if self._is_online() and self.system_devices else "disabled")
        self.offline_mode.config(state="normal")
        self.online_mode.config(state="normal")
        self._restart_monitor()

    def open_existing(self) -> None:
        if self.recorder is not None or self.processing:
            return
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        selected = filedialog.askopenfilename(
            title="选择要重新处理的录音",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("WAV 录音", "*.wav"), ("所有文件", "*.*")],
        )
        if not selected:
            return
        audio_path = Path(selected)
        meeting_dir = audio_path.parent
        if (meeting_dir / "会议纪要.md").exists() and not messagebox.askyesno(
            "已有纪要",
            f"{meeting_dir.name} 里已经有会议纪要.md，重新处理会覆盖它和逐字稿.txt。\n\n继续吗？",
        ):
            return
        title = safe_name(re.sub(r"^\d{8}-\d{6}-", "", meeting_dir.name) or audio_path.stem)
        self._stop_monitor()
        self._start_processing(meeting_dir, audio_path, title)

    def _start_processing(self, meeting_dir: Path, audio_path: Path, title: str) -> None:
        self.meeting_dir = meeting_dir
        self.audio_path = audio_path
        self.meeting_title = title
        self.processing = True
        self._cancel.clear()
        self.diarize_now = bool(self.diarize_enabled.get()) and diarization_available()
        self.stage_count = 3 if self.diarize_now else 2
        self._stop_monitor()
        self.start_button.config(state="disabled")
        self.existing_button.config(state="disabled")
        self.title_entry.config(state="disabled")
        self.device_box.config(state="disabled")
        self.system_box.config(state="disabled")
        self.offline_mode.config(state="disabled")
        self.online_mode.config(state="disabled")
        self.stop_button.config(text="停止处理", state="normal", command=self.cancel_processing)
        self.progress.config(mode="determinate", value=0)
        threading.Thread(
            target=self._process_meeting, args=(meeting_dir, audio_path, title), daemon=True
        ).start()

    def cancel_processing(self) -> None:
        if not self.processing or self._cancel.is_set():
            return
        if not messagebox.askyesno(
            "停止处理", "已经转写出来的部分会保留在逐字稿.txt，剩下的需要以后重新处理。\n\n确定停止吗？"
        ):
            return
        self._cancel.set()
        self.stop_button.config(state="disabled")
        self.status_label.config(text="正在停止，请等当前这一段转写完……")

    def _post(self, callback, *args) -> None:
        try:
            self.after(0, callback, *args)
        except Exception:
            pass

    def _say(self, text: str) -> None:
        self._post(self.status_label.config, {"text": text})

    def _show_progress(self, done: float, total: float, count: int) -> None:
        percent = min(100.0, done / total * 100) if total > 0 else 0.0
        self.progress.config(value=percent)
        elapsed = time.monotonic() - self._transcribe_started
        left = ""
        if done > 30 and elapsed > 10:
            remaining = (total - done) / (done / elapsed)
            left = f"，预计还要{format_minutes_left(remaining)}"
        self.status_label.config(
            text=f"阶段 1/{self.stage_count} 转写中 {percent:.0f}%（{timestamp(done)} / {timestamp(total)}），"
            f"已识别 {count} 句{left}。逐字稿正在实时保存，可随时“停止处理”。"
        )

    def _report_progress(self, done: float, total: float, count: int) -> None:
        self._post(self._show_progress, done, total, count)

    def _diarize_progress(self, processed: int, total: int, _arg=None) -> int:
        percent = processed / total * 100 if total else 0.0
        self._post(self._show_diarize_progress, percent)
        return 0

    def _show_diarize_progress(self, percent: float) -> None:
        self.progress.config(value=percent)
        self.status_label.config(
            text=f"阶段 2/3 正在识别谁在说话 {percent:.0f}%（这一步大约要录音时长的十分之一）。"
        )

    def _track_log(self, recorder: Recorder) -> list[str]:
        try:
            device = sd.query_devices(recorder.device if recorder.device is not None else sd.default.device[0])
            name = f"{device['name']}  [{sd.query_hostapis(device['hostapi'])['name']}]"
        except Exception:
            name = "未知设备"
        lines = [
            f"—— {recorder.label}（{recorder.path.name if recorder.path else '未写文件'}）",
            f"设备：{name}",
            f"录到的音频长度：{timestamp(recorder.recorded_seconds)}",
            f"程序运行时长：{timestamp(time.monotonic() - recorder.started_at)}",
            f"最大音量峰值：{recorder.max_peak}/32768（{peak_to_dbfs(recorder.max_peak):.0f} dBFS）",
            f"自动重连次数：{recorder.restarts}",
            f"输入溢出：{'有' if recorder.overflowed else '无'}",
        ]
        if recorder.preroll_seconds > 0:
            lines.append(f"含点“开始录音”之前的预录：{recorder.preroll_seconds:.0f} 秒")
        lines.extend(recorder.events)
        return lines

    def _write_recording_log(
        self,
        recorder: Recorder,
        backup: Recorder | None = None,
        system: Recorder | None = None,
    ) -> None:
        if self.meeting_dir is None:
            return
        lines = self._track_log(recorder)
        lines.append("")
        if backup is not None:
            lines.extend(self._track_log(backup))
        else:
            lines.append("—— 备份录音：没有可用的第二个输入设备，本次只录了一路")
        if system is not None:
            lines.append("")
            lines.extend(self._track_log(system))
        try:
            (self.meeting_dir / "录音日志.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _process_meeting(self, meeting_dir: Path, audio_path: Path, title: str) -> None:
        transcript_path = meeting_dir / "逐字稿.txt"
        boosted_path: Path | None = None
        diarize_copy = meeting_dir / "_分离副本.wav"
        stages = self.stage_count
        try:
            total_seconds, peak = wav_stats(audio_path)
            source_path = audio_path
            gain = 1.0
            if SILENCE_PEAK <= peak < BOOST_TRIGGER_PEAK:
                self._say("录音音量偏小，正在生成放大后的转写副本……")
                boosted_path = meeting_dir / "_转写副本.wav"
                gain = boost_wav(audio_path, boosted_path, peak)
                if gain > 1.0:
                    source_path = boosted_path

            self._say(f"正在加载转写模型 {WHISPER_MODEL}（首次使用需要下载，请保持联网）……")
            from faster_whisper import WhisperModel

            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            segments, _ = model.transcribe(
                str(source_path),
                language="zh",
                vad_filter=True,
                vad_parameters={"threshold": 0.35, "min_silence_duration_ms": 700, "speech_pad_ms": 300},
                beam_size=5,
                condition_on_previous_text=False,
                initial_prompt="以下是中文会议录音。请使用简体中文准确转写，保留数字、专有名词和语气，不要猜测听不清的内容。",
            )
            timed_lines: list[str] = []
            plain_lines: list[str] = []
            spans: list[tuple[float, float]] = []
            confidence_scores: list[float] = []
            cancelled = False
            dropped = 0
            repeats = 0
            last_text = ""
            self._transcribe_started = time.monotonic()
            last_report = 0.0
            with open(transcript_path, "w", encoding="utf-8") as handle:
                handle.write("转写进行中，本文件会随转写实时追加……\n\n")
                handle.flush()
                for segment in segments:
                    if self._cancel.is_set():
                        cancelled = True
                        break
                    text = segment.text.strip()
                    if text:
                        repeats = repeats + 1 if text == last_text else 0
                        last_text = text
                        if is_hallucination(text) or repeats >= MAX_REPEATS:
                            dropped += 1
                        else:
                            line = f"[{timestamp(segment.start)}] {text}"
                            timed_lines.append(line)
                            plain_lines.append(text)
                            spans.append((segment.start, segment.end))
                            confidence_scores.append(segment.avg_logprob)
                            handle.write(line + "\n")
                            handle.flush()
                    now = time.monotonic()
                    if now - last_report > 0.5:
                        last_report = now
                        self._report_progress(segment.end, total_seconds, len(timed_lines))

            quality = transcription_quality(confidence_scores)
            notes = [f"转写质量：{quality}"]
            if gain > 1.0:
                notes.append(f"转写前已自动放大音量 {gain:.1f} 倍（原始录音未改动）")
            if dropped:
                notes.append(f"已自动过滤 {dropped} 行疑似广告语或复读幻觉")
            if cancelled:
                notes.append("⚠ 本次转写被手动停止，以下只是录音的前一部分")

            del segments
            del model
            gc.collect()

            speakers = 0
            if plain_lines and not cancelled and self.diarize_now:
                try:
                    self._say(f"阶段 2/{stages} 正在识别谁在说话……")
                    turns = diarize(diarization_source(source_path, diarize_copy), progress=self._diarize_progress)
                    labels = label_speakers(spans, turns)
                    speakers = len({label for label in labels if label is not None})
                    if speakers >= 2:
                        timed_lines = [
                            f"[{timestamp(start)}] {speaker_name(label)}：{text}"
                            for (start, _end), label, text in zip(spans, labels, plain_lines)
                        ]
                        notes.append(f"说话人分离：共 {speakers} 位（机器自动判断，可能有误）")
                    else:
                        notes.append("说话人分离：只听出一位说话人，未标注")
                except Exception as error:
                    notes.append(f"说话人分离没有成功：{error}")

            transcript = "\n".join(timed_lines)
            transcript_path.write_text("\n".join(notes) + "\n\n" + transcript + "\n", encoding="utf-8")

            if cancelled:
                self._post(self._processing_done, None, "已停止。逐字稿保留了已经转写出来的部分。")
                return
            if not plain_lines:
                raise RuntimeError("没有识别到有效语音，请试听原始录音并查看录音日志.txt")

            ready, reason = ollama_ready()
            self._post(self._begin_minutes_phase)
            if ready:
                self._say(f"阶段 {stages}/{stages} 正在用 {OLLAMA_MODEL} 生成会议纪要……")
                try:
                    minutes = generate_minutes(
                        transcript,
                        title,
                        audio_path,
                        quality,
                        speakers,
                        lambda index, count: self._say(
                            f"阶段 {stages}/{stages} 正在生成会议纪要：第 {index}/{count} 步……"
                        ),
                    )
                    reason = ""
                except Exception as error:
                    reason = str(error)
                    minutes = fallback_minutes("\n".join(plain_lines), title, audio_path)
                finally:
                    stop_ollama_model(OLLAMA_MODEL)
            else:
                minutes = fallback_minutes("\n".join(plain_lines), title, audio_path)
            (meeting_dir / "会议纪要.md").write_text(minutes, encoding="utf-8")
            done_note = f"会议纪要是简易兜底版：{reason}" if reason else ""
            self._post(self._processing_done, None, done_note)
        except Exception as error:
            try:
                (meeting_dir / "处理失败说明.txt").write_text(
                    f"原始录音已安全保存：{audio_path}\n处理失败：{error}\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
            self._post(self._processing_done, str(error), "")
        finally:
            for temporary in (boosted_path, diarize_copy):
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except Exception:
                        pass

    def _begin_minutes_phase(self) -> None:
        self.progress.config(mode="indeterminate")
        self.progress.start(12)

    def _processing_done(self, error: str | None, note: str = "") -> None:
        self.processing = False
        self._cancel.clear()
        self.progress.stop()
        self.progress.config(mode="determinate", value=0)
        self._restore_ready_controls()
        self.stop_button.config(text="结束并生成纪要", command=self.stop_recording, state="disabled")
        if error:
            self.status_label.config(text=f"原始录音已保存，但自动处理失败：{error}")
            messagebox.showwarning("处理未完成", "原始录音没有丢失。请打开结果文件夹查看处理失败说明。")
        elif note:
            self.status_label.config(text=note)
            messagebox.showinfo("处理结束", note)
        else:
            self.status_label.config(text="完成：原始录音、逐字稿和会议纪要均已保存。")
            messagebox.showinfo("处理完成", "已生成逐字稿和会议纪要。")

    def open_results(self) -> None:
        target = self.meeting_dir or OUTPUT_DIR
        target.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer.exe", str(target)])

    def on_close(self) -> None:
        if self.recorder is not None:
            if not messagebox.askyesno("正在录音", "关闭程序会立即停止录音。确定关闭吗？"):
                return
            recorder = self.recorder
            backup = self.backup
            system = self.system_recorder
            self.recorder = None
            self.backup = None
            self.system_recorder = None
            try:
                stop_errors = stop_recorders_safely(recorder, backup, system)
                self._write_recording_log(recorder, backup, system)
                if stop_errors:
                    messagebox.showwarning(
                        "录音收尾异常",
                        "所有音轨均已尝试停止，但部分文件可能不完整：\n\n" + "\n".join(stop_errors),
                    )
                if (
                    system is not None
                    and system.error is None
                    and system.path is not None
                    and self.audio_path is not None
                ):
                    mic_path = choose_microphone_recording(recorder, backup)
                    chosen_recorder = backup if backup is not None and mic_path == backup.path else recorder
                    self._mix_complete_recording(chosen_recorder, mic_path, system)
            except Exception as error:
                messagebox.showwarning(
                    "录音收尾未完成",
                    f"已尽力保留所有原始音轨，但关闭时的最终校验或合成失败。\n\n{error}",
                )
            finally:
                prevent_system_sleep(False)
        elif self.processing:
            if not messagebox.askyesno(
                "正在处理",
                "原始录音和已经转写出来的逐字稿都已保存，但关闭会中断剩下的处理。\n"
                "以后可以用“处理已有录音…”重新来一次。\n\n确定关闭吗？",
            ):
                return
            self._cancel.set()
        self._stop_monitor()
        self.destroy()


if __name__ == "__main__":
    app = MeetingApp()
    if "--autostart" in sys.argv:
        app.after(1_500, app.start_recording)
    app.mainloop()
