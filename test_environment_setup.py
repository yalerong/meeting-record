import io
import json
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

import download_models
from environment_setup import (
    SetupCancelled, WHISPER_FILES, ollama_pull, prepare_whisper, whisper_ready,
)


def complete_whisper(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in WHISPER_FILES:
        value = b"weights" if name == "model.bin" else (
            b'["token"]' if name == "vocabulary.json" else b'{"test": 1}'
        )
        (path / name).write_bytes(value)


class Response(io.BytesIO):
    def __init__(self, data, expected=None):
        super().__init__(data)
        self.headers = {"Content-Length": str(expected if expected is not None else len(data))}


class WhisperSetupTest(unittest.TestCase):
    def test_local_readiness_requires_nonempty_valid_files_without_importing_whisper(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            with mock.patch.dict(sys.modules, {"faster_whisper": None}):
                self.assertFalse(whisper_ready(path))
                complete_whisper(path)
                self.assertTrue(whisper_ready(path))
                (path / "tokenizer.json").unlink()
                self.assertFalse(whisper_ready(path))
                (path / "tokenizer.json").write_text("{", encoding="utf-8")
                self.assertFalse(whisper_ready(path))

    def test_complete_local_model_skips_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            complete_whisper(path)
            with mock.patch.dict(sys.modules, {"faster_whisper": None}):
                self.assertEqual(prepare_whisper(path), path)

    def test_download_uses_official_output_dir_and_validates_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model"
            utils = ModuleType("faster_whisper.utils")
            utils.download_model = mock.Mock(side_effect=lambda *args, **kwargs: complete_whisper(path))
            messages = []
            with mock.patch.dict(sys.modules, {"faster_whisper.utils": utils}):
                self.assertEqual(prepare_whisper(path, messages.append), path)
            utils.download_model.assert_called_once_with("large-v3-turbo", output_dir=str(path))
            self.assertIn("取消将在当前下载返回后生效", messages[0])
            self.assertIn("完成", messages[-1])

    def test_incomplete_download_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            utils = ModuleType("faster_whisper.utils")
            utils.download_model = mock.Mock()
            messages = []
            with mock.patch.dict(sys.modules, {"faster_whisper.utils": utils}):
                with self.assertRaisesRegex(RuntimeError, "未完整"):
                    prepare_whisper(Path(temporary), messages.append)
            self.assertFalse(any(message == "转写模型下载完成。" for message in messages))

    def test_download_cancellation_is_checked_at_documented_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            cancelled = []
            utils = ModuleType("faster_whisper.utils")

            def download(*args, **kwargs):
                complete_whisper(path)
                cancelled.append(True)

            utils.download_model = mock.Mock(side_effect=download)
            messages = []
            with mock.patch.dict(sys.modules, {"faster_whisper.utils": utils}):
                with self.assertRaises(SetupCancelled):
                    prepare_whisper(path, messages.append, lambda: bool(cancelled))
            self.assertTrue(whisper_ready(path))
            self.assertNotEqual(messages[-1], "转写模型下载完成。")


class OllamaSetupTest(unittest.TestCase):
    def test_streams_real_progress_and_requires_success(self):
        body = b'{"status":"pulling","total":100,"completed":25}\n{"status":"success"}\n'
        messages = []
        with mock.patch("environment_setup.request.urlopen", return_value=Response(body)) as opened:
            ollama_pull("qwen2.5:3b", messages.append)
        req = opened.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/pull")
        self.assertEqual(json.loads(req.data), {"model": "qwen2.5:3b", "stream": True})
        self.assertTrue(any("25%" in message for message in messages))

    def test_server_error_or_early_eof_is_a_failure(self):
        for body in (b'{"error":"not found"}\n', b'{"status":"pulling"}\n', b'[]\n'):
            with self.subTest(body=body):
                with mock.patch("environment_setup.request.urlopen", return_value=Response(body)):
                    with self.assertRaises(RuntimeError):
                        ollama_pull("test")

    def test_cancel_between_events_closes_stream(self):
        response = Response(b'{"status":"pulling"}\n{"status":"success"}\n')
        cancelled = []

        def progress(message):
            if message.endswith("：pulling"):
                cancelled.append(True)

        with mock.patch("environment_setup.request.urlopen", return_value=response):
            with self.assertRaises(SetupCancelled):
                ollama_pull("test", progress, lambda: bool(cancelled))
        self.assertTrue(response.closed)

    def test_cancel_before_request_makes_no_network_call(self):
        with mock.patch("environment_setup.request.urlopen") as opened:
            with self.assertRaises(SetupCancelled):
                ollama_pull("test", cancel=lambda: True)
        opened.assert_not_called()


class SpeakerSetupTest(unittest.TestCase):
    def test_download_publishes_only_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "model.onnx"
            body = b"x" * 2048

            def progress(message):
                if not message.startswith("下载完成"):
                    self.assertFalse(destination.exists())

            with mock.patch("download_models.urllib.request.urlopen", return_value=Response(body)):
                download_models.fetch(download_models.EMBEDDING_URL, destination, progress)
            self.assertEqual(destination.read_bytes(), body)
            self.assertFalse(destination.with_suffix(".onnx.part").exists())

    def test_short_or_cancelled_download_does_not_publish_model(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled), tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / "model.onnx"
                cancellation = []

                def progress(message):
                    if "MB" in message and cancelled:
                        cancellation.append(True)

                response = Response(b"x" * 2048, expected=4096)
                with mock.patch("download_models.urllib.request.urlopen", return_value=response):
                    with self.assertRaises(RuntimeError):
                        download_models.fetch(
                            download_models.EMBEDDING_URL, destination, progress,
                            lambda: bool(cancellation),
                        )
                self.assertFalse(destination.exists())
                self.assertFalse(destination.with_suffix(".onnx.part").exists())

    def test_unknown_download_url_is_rejected(self):
        with mock.patch("download_models.urllib.request.urlopen") as opened:
            with self.assertRaises(ValueError):
                download_models.fetch("https://example.com/model", Path("unused"))
        opened.assert_not_called()

    def test_preparation_extracts_to_staging_then_publishes_models(self):
        model = os.urandom(2048)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
            info = tarfile.TarInfo("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
            info.size = len(model)
            archive.addfile(info, io.BytesIO(model))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("download_models.urllib.request.urlopen", side_effect=[
                Response(buffer.getvalue()), Response(b"e" * 2048),
            ]):
                download_models.prepare_models(root)
            self.assertEqual(
                (root / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx").read_bytes(), model,
            )
            self.assertEqual(
                (root / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx").stat().st_size, 2048,
            )
            self.assertFalse(list(root.glob(".segmentation-*")))

    def test_archive_path_traversal_is_rejected(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
            info = tarfile.TarInfo("../../escape.onnx")
            body = os.urandom(2048)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            with mock.patch("download_models.urllib.request.urlopen", return_value=Response(buffer.getvalue())):
                with self.assertRaises(tarfile.OutsideDestinationError):
                    download_models.prepare_models(root)
            self.assertFalse((root.parent / "escape.onnx").exists())


if __name__ == "__main__":
    unittest.main()
