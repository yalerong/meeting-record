import json
import queue
import threading
import tempfile
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest import mock

import app


class OllamaStreamingTest(unittest.TestCase):
    def test_cold_start_first_token_can_take_more_than_thirty_seconds(self):
        # Scale socket timeouts so a real delayed HTTP response tests a >30s
        # cold start without making the regression suite wait that long.
        class DelayedOllama(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self.send_response(200)
                self.end_headers()
                self.wfile.flush()
                threading.Event().wait(0.25)
                try:
                    self.wfile.write(b'{"response":"AI minutes","done":true}\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *_args):
                pass

        with HTTPServer(("127.0.0.1", 0), DelayedOllama) as server:
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
            thread.start()
            real_urlopen = app.request.urlopen

            def local_service(req, *, timeout):
                redirected = app.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/generate",
                    data=req.data, headers=dict(req.header_items()), method="POST",
                )
                return real_urlopen(redirected, timeout=timeout / 200)

            try:
                with mock.patch("app.request.urlopen", side_effect=local_service):
                    self.assertEqual(app.run_ollama("long meeting"), "AI minutes")
            finally:
                server.shutdown()
                thread.join(timeout=2)

    def response(self, *items):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.__iter__.return_value = iter(
            (json.dumps(item).encode("utf-8") + b"\n") for item in items
        )
        return response

    def test_stream_updates_before_final_result_and_requests_streaming(self):
        updates = []
        response = self.response(
            {"response": "会议", "done": False},
            {"response": "纪要", "done": True},
        )
        with mock.patch("app.request.urlopen", return_value=response) as open_url:
            result = app.run_ollama("prompt", on_text=updates.append)
        self.assertEqual(result, "会议纪要")
        self.assertEqual(updates, ["会议", "纪要"])
        payload = json.loads(open_url.call_args.args[0].data)
        self.assertTrue(payload["stream"])

    def test_cancel_prevents_request(self):
        cancel = threading.Event()
        cancel.set()
        with mock.patch("app.request.urlopen") as open_url:
            with self.assertRaises(app.ProcessingCancelled):
                app.run_ollama("prompt", cancel=cancel)
        open_url.assert_not_called()

    def test_cancel_during_stream_closes_response(self):
        cancel = threading.Event()
        response = self.response(
            {"response": "partial", "done": False},
            {"response": "unwanted", "done": True},
        )
        with mock.patch("app.request.urlopen", return_value=response):
            with self.assertRaises(app.ProcessingCancelled):
                app.run_ollama("prompt", cancel=cancel, on_text=lambda _: cancel.set())
        response.__exit__.assert_called_once()

    def test_incomplete_stream_is_failure(self):
        with mock.patch("app.request.urlopen", return_value=self.response({"response": "partial"})):
            with self.assertRaisesRegex(RuntimeError, "中断"):
                app.run_ollama("prompt")

    def test_stream_error_is_visible(self):
        with mock.patch("app.request.urlopen", return_value=self.response({"error": "model missing"})):
            with self.assertRaisesRegex(RuntimeError, "model missing"):
                app.run_ollama("prompt")

    def test_minutes_does_not_continue_after_cancelled_chunk(self):
        cancel = threading.Event()

        def generate(*_args, **_kwargs):
            cancel.set()
            return "first chunk"

        with (
            mock.patch("app.split_text", return_value=["one", "two"]),
            mock.patch("app.pick_chunk_model", return_value=app.OLLAMA_MODEL),
            mock.patch("app.run_ollama", side_effect=generate) as run,
            self.assertRaises(app.ProcessingCancelled),
        ):
            app.generate_minutes("text", "title", app.Path("audio.wav"), cancel=cancel)
        self.assertEqual(run.call_count, 1)


class ConsoleDispatchTest(unittest.TestCase):
    def test_cancel_requested_before_completion_callback_is_not_lost(self):
        target = SimpleNamespace(
            _cancel=threading.Event(), progress=mock.Mock(), _set_task_state=mock.Mock(),
            _close_when_idle=False, _restore_ready_controls=mock.Mock(), stop_button=mock.Mock(),
            stop_recording=mock.Mock(), status_label=mock.Mock(),
        )
        target._cancel.set()
        with mock.patch("app.messagebox.showinfo"):
            app.MeetingApp._processing_done(target, None)
        self.assertEqual(target._set_task_state.call_args.args[0], "cancelled")
        self.assertFalse(target._cancel.is_set())

    def test_worker_post_only_enqueues_and_never_calls_tk(self):
        target = SimpleNamespace(_events=queue.Queue(), _closing=False)
        callback = mock.Mock()
        app.MeetingApp._post(target, callback, "value")
        callback.assert_not_called()
        queued_callback, args = target._events.get_nowait()
        self.assertIs(queued_callback, callback)
        self.assertEqual(args, ("value",))

    def test_closed_window_drops_worker_updates(self):
        target = SimpleNamespace(_events=queue.Queue(), _closing=True)
        app.MeetingApp._post(target, mock.Mock())
        self.assertTrue(target._events.empty())

    def test_diarization_callback_obeys_cancel_signal(self):
        cancel = threading.Event()
        cancel.set()
        target = SimpleNamespace(_cancel=cancel, _post=mock.Mock(), _show_diarize_progress=mock.Mock())
        # sherpa-onnx documents a nonzero callback result as a request to stop.
        self.assertNotEqual(app.MeetingApp._diarize_progress(target, 1, 10), 0)

    def test_finalization_runs_without_tk_and_posts_completion(self):
        recorder = mock.Mock(path=Path("original.wav"), error=None, max_peak=1000,
                             expected_seconds=1.0, overflowed=False)
        target = SimpleNamespace(_write_recording_log=mock.Mock(), _post=mock.Mock(), _recording_saved=mock.Mock())
        with mock.patch("app.verify_wav_file"), mock.patch("app.choose_microphone_recording", return_value=recorder.path):
            app.MeetingApp._finalize_recording(target, recorder, None, None)
        recorder.stop.assert_called_once()
        target._post.assert_called_once_with(target._recording_saved, recorder.path, [], [], None)

    def test_required_track_failure_blocks_automatic_processing(self):
        recorder = mock.Mock(path=Path("original.wav"), error="disk full")
        target = SimpleNamespace(_write_recording_log=mock.Mock(), _post=mock.Mock(), _recording_saved=mock.Mock())
        with mock.patch("app.verify_wav_file") as verify, mock.patch("app.logging.exception"):
            app.MeetingApp._finalize_recording(target, recorder, None, None)
        verify.assert_not_called()
        self.assertIn("disk full", target._post.call_args.args[-1])


class ProcessingLifecycleTest(unittest.TestCase):
    def test_interrupted_online_recording_can_retry_raw_track_before_mix_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            tasks = app.TaskStore(root)
            target = SimpleNamespace(
                recorder=None, processing=False, saving=False, _setup_busy=False,
                tasks=tasks, settings={}, meeting_mode=mock.Mock(), preview=mock.Mock(),
                _is_online=lambda: True, _selected_device=lambda: 1,
                _selected_system_device=lambda: 2, _take_preroll=lambda **_: (b"", 0.0),
                _start_backup=mock.Mock(return_value=None), _set_task_state=mock.Mock(),
                _restart_monitor=mock.Mock(), _log_event=mock.Mock(),
            )
            for name in ("title_entry", "status_label", "start_button", "existing_button", "stop_button",
                         "device_box", "system_box", "offline_mode", "online_mode", "refresh_devices_button"):
                setattr(target, name, mock.Mock())
            target.title_entry.get.return_value = "online meeting"

            def recorder(path, **_kwargs):
                def write_raw_audio():
                    with wave.open(str(path), "wb") as writer:
                        writer.setnchannels(1)
                        writer.setsampwidth(2)
                        writer.setframerate(16000)
                        writer.writeframes(b"\x00\x10" * 1600)
                return SimpleNamespace(path=path, start=write_raw_audio)

            with (
                mock.patch("app.OUTPUT_DIR", root),
                mock.patch("app.Recorder", side_effect=recorder),
                mock.patch("app.input_capture_settings", return_value=(16000, 1, None)),
                mock.patch("app.shutil.disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)),
                mock.patch("app.save_json"),
                mock.patch("app.prevent_system_sleep", return_value=True),
            ):
                app.MeetingApp.start_recording(target)

            restarted = app.TaskStore(root)
            self.assertEqual(restarted.recover_interrupted(), 1)
            recovered = restarted.history()[0]
            self.assertEqual(recovered["state"], "interrupted")
            self.assertEqual(Path(recovered["audio_path"]), target.recorder.path)
            self.assertTrue(Path(recovered["audio_path"]).is_file())
            self.assertFalse(target.audio_path.exists(), "Mixed output must still be absent before finalization")
            raw_path = target.recorder.path
            target.recorder = None
            target._selected_history = lambda: recovered
            target._process_existing_path = mock.Mock()
            app.MeetingApp._retry_history(target)
            target._process_existing_path.assert_called_once_with(raw_path, "online meeting")

    def test_installed_audio_decoder_can_read_pcm_wave(self):
        # Use the real native decoder: import-only smoke tests missed PyAV 19's API break.
        from faster_whisper.audio import decode_audio
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "input.wav"
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                writer.writeframes(b"\x00\x10" * 16000)
            decoded = decode_audio(str(audio), sampling_rate=16000)
        self.assertEqual(len(decoded), 16000)
        self.assertGreater(float(decoded.max()), 0)

    def run_processing(self, directory, *, cancel_at="", diarize_now=False):
        """Exercise the actual orchestrator with small generated audio and fake engines."""
        audio = directory / "audio.wav"
        with wave.open(str(audio), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(b"\x00\x20" * 16000)
        target = object.__new__(app.MeetingApp)
        target.stage_count = 3 if diarize_now else 2
        target.diarize_now = diarize_now
        target._cancel = threading.Event()
        target.preview = mock.Mock()
        target._post = mock.Mock()
        target._say = mock.Mock()
        segment = SimpleNamespace(text="确认下周交付", start=0, end=1, avg_logprob=-0.1)
        fake_model = mock.Mock()
        fake_model.transcribe.return_value = (iter([segment]), None)

        def split(*_args, **_kwargs):
            if cancel_at == "diarization":
                target._cancel.set()
            return []

        def summarize(*_args, **_kwargs):
            if cancel_at == "minutes":
                raise app.ProcessingCancelled("cancel")
            return "# 会议纪要\n已确认交付。"

        with (
            mock.patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=mock.Mock(return_value=fake_model))}),
            mock.patch("app.diarize", side_effect=split),
            mock.patch("app.diarization_source", return_value=audio),
            mock.patch("app.ollama_ready", return_value=(True, "")),
            mock.patch("app.generate_minutes", side_effect=summarize) as minutes,
        ):
            target._process_meeting(directory, audio, "验收会议")
        return target, minutes

    def test_cancelled_diarization_preserves_transcript_and_skips_minutes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target, minutes = self.run_processing(directory, cancel_at="diarization", diarize_now=True)
            self.assertIn("确认下周交付", (directory / "逐字稿.txt").read_text(encoding="utf-8"))
            self.assertFalse((directory / "会议纪要.md").exists())
            minutes.assert_not_called()
            self.assertTrue(target._post.call_args.args[-1])

    def test_cancelled_minutes_does_not_publish_fallback_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target, _ = self.run_processing(directory, cancel_at="minutes")
            self.assertFalse((directory / "会议纪要.md").exists())
            self.assertFalse((directory / "处理失败说明.txt").exists())
            self.assertTrue(target._post.call_args.args[-1])

    def test_success_writes_results_and_posts_completed_callback(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            target, _ = self.run_processing(directory)
            self.assertIn("已确认交付", (directory / "会议纪要.md").read_text(encoding="utf-8"))
            self.assertIsNone(target._post.call_args.args[1])
            self.assertEqual(target._post.call_args.args[2], "")


if __name__ == "__main__":
    unittest.main()
