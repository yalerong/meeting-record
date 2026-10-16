"""Persistent desktop data and task history, independent of the GUI and audio stack."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    meetings_dir: Path
    models_dir: Path
    logs_dir: Path
    config_path: Path
    resource_dir: Path

    @property
    def legacy_meetings_dir(self) -> Path:
        """Old source-tree recordings stay in place; the GUI can offer to open them."""
        return self.resource_dir / "meetings"

    def model_path(self, relative: str | Path) -> Path:
        persistent = self.models_dir / relative
        bundled = self.resource_dir / "models" / relative
        return bundled if not persistent.exists() and bundled.exists() else persistent


def get_app_paths(source_dir: Path | None = None) -> AppPaths:
    resource_dir = Path(
        getattr(sys, "_MEIPASS", source_dir or Path(__file__).resolve().parent)
    ).resolve()
    local_data = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    data_dir = Path(os.environ.get("MEETING_RECORD_HOME") or local_data / "MeetingRecord").resolve()
    paths = AppPaths(
        data_dir=data_dir,
        meetings_dir=data_dir / "meetings",
        models_dir=data_dir / "models",
        logs_dir=data_dir / "logs",
        config_path=data_dir / "config.json",
        resource_dir=resource_dir,
    )
    for directory in (paths.meetings_dir, paths.models_dir, paths.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as source:
            return json.load(source)
    except FileNotFoundError:
        return default


def save_json(path: Path, data: Any) -> None:
    """Replace a complete JSON document; a failed write preserves the last version."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


TASK_STATES = frozenset({
    "recording", "saving", "transcribing", "diarizing", "summarizing",
    "completed", "cancelled", "failed", "interrupted",
})
TERMINAL_STATES = frozenset({"completed", "cancelled", "failed", "interrupted"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class TaskStore:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.errors: list[str] = []
        self.recovery_errors: list[str] = []
        self._lock = threading.RLock()

    def create(
        self, title: str, audio_path: Path | None = None,
        meeting_dir: Path | None = None, state: str = "recording",
    ) -> dict[str, Any]:
        if state not in TASK_STATES:
            raise ValueError(f"Unknown task state: {state}")
        directory = Path(meeting_dir or self.output_dir / uuid.uuid4().hex).resolve()
        with self._lock:
            if (directory / "task.json").exists():
                raise FileExistsError(f"Task already exists: {directory}")
            task = {
                "title": title, "state": state, "stage": state, "message": "",
                "updated_at": _now(), "meeting_dir": str(directory),
                "audio_path": str(Path(audio_path).resolve()) if audio_path else "",
            }
            save_json(directory / "task.json", task)
            return task

    def update(self, meeting_dir: Path, **fields: Any) -> dict[str, Any]:
        directory = Path(meeting_dir).resolve()
        with self._lock:
            task = self._read(directory / "task.json")
            if "state" in fields and fields["state"] not in TASK_STATES:
                raise ValueError(f"Unknown task state: {fields['state']}")
            if "audio_path" in fields:
                fields["audio_path"] = str(fields["audio_path"]) if fields["audio_path"] else ""
            task.update(fields)
            task["meeting_dir"] = str(directory)
            task["updated_at"] = _now()
            save_json(directory / "task.json", task)
            return task

    @staticmethod
    def _read(path: Path) -> dict[str, Any]:
        task = load_json(path)
        if not isinstance(task, dict) or task.get("state") not in TASK_STATES:
            raise ValueError(f"Invalid task record: {path}")
        if not isinstance(task.get("updated_at"), str) or not isinstance(task.get("title"), str):
            raise ValueError(f"Invalid task record: {path}")
        updated = datetime.fromisoformat(task["updated_at"])
        if updated.utcoffset() is None:
            raise ValueError(f"Task timestamp has no timezone: {path}")
        task["meeting_dir"] = str(path.parent.resolve())
        return task

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            self.errors = []
            tasks = []
            for path in self.output_dir.glob("*/task.json"):
                try:
                    tasks.append(self._read(path))
                except (OSError, ValueError, TypeError) as exc:
                    self.errors.append(f"{path}: {exc}")
            return sorted(
                tasks, key=lambda task: datetime.fromisoformat(task["updated_at"]), reverse=True,
            )

    def recover_interrupted(self) -> int:
        """Call once after acquiring the application's single-instance lock."""
        with self._lock:
            self.recovery_errors = []
            unfinished = [task for task in self.history() if task["state"] not in TERMINAL_STATES]
            recovered = 0
            for task in unfinished:
                try:
                    self.update(
                        Path(task["meeting_dir"]), state="interrupted",
                        message="上次运行意外结束，已有文件已保留；可选择录音重新处理。",
                    )
                except OSError as exc:
                    self.recovery_errors.append(f"{task['meeting_dir']}: {exc}")
                else:
                    recovered += 1
            return recovered
