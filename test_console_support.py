import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from console_support import TaskStore, get_app_paths, load_json, save_json


class AppPathsTest(unittest.TestCase):
    def test_frozen_bundle_is_never_the_default_data_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "_MEI123"
            bundle.mkdir()
            with mock.patch.dict(os.environ, {"LOCALAPPDATA": str(root / "local")}, clear=True):
                with mock.patch("sys._MEIPASS", str(bundle), create=True):
                    paths = get_app_paths()
            self.assertEqual(paths.data_dir, root / "local" / "MeetingRecord")
            self.assertEqual(paths.resource_dir, bundle)
            self.assertTrue(paths.meetings_dir.is_dir())
            self.assertEqual(list(bundle.iterdir()), [])

    def test_home_override_and_existing_source_models_are_read_only_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            model = source / "models" / "speaker.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"existing")
            with mock.patch.dict(os.environ, {"MEETING_RECORD_HOME": str(root / "data")}):
                paths = get_app_paths(source)
            self.assertEqual(paths.data_dir, root / "data")
            self.assertEqual(paths.model_path("speaker.onnx"), model)
            self.assertFalse(paths.legacy_meetings_dir.exists())
            local_model = paths.models_dir / "speaker.onnx"
            local_model.write_bytes(b"new")
            self.assertEqual(paths.model_path("speaker.onnx"), local_model)
            self.assertEqual(model.read_bytes(), b"existing")


class AtomicJsonTest(unittest.TestCase):
    def test_failed_replace_keeps_previous_complete_document_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_json(path, {"value": "old"})
            with mock.patch("console_support.os.replace", side_effect=PermissionError("busy")):
                with self.assertRaises(PermissionError):
                    save_json(path, {"value": "new"})
            self.assertEqual(load_json(path), {"value": "old"})
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_invalid_serialization_keeps_previous_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            save_json(path, {"value": "old"})
            with self.assertRaises(TypeError):
                save_json(path, {"value": object()})
            self.assertEqual(load_json(path), {"value": "old"})
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_json(path, {})
            self.assertEqual(load_json(path.parent / "missing", {}), {})


class TaskStoreTest(unittest.TestCase):
    def test_external_audio_has_persistent_history_without_moving_the_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "external" / "meeting.wav"
            audio.parent.mkdir()
            audio.write_bytes(b"audio")
            store = TaskStore(root / "meetings")
            task = store.create("外部会议", audio, state="transcribing")
            restored = TaskStore(root / "meetings").history()
            self.assertEqual(restored, [task])
            self.assertEqual(restored[0]["audio_path"], str(audio))
            self.assertTrue(Path(task["meeting_dir"]).is_relative_to(root / "meetings"))
            self.assertEqual(audio.read_bytes(), b"audio")

    def test_restart_marks_only_unfinished_tasks_interrupted(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary))
            for state in ("recording", "saving", "transcribing", "diarizing", "summarizing",
                          "completed", "cancelled", "failed", "interrupted"):
                store.create(state, state=state)
            restarted = TaskStore(Path(temporary))
            self.assertEqual(restarted.recover_interrupted(), 5)
            states = {task["title"]: task["state"] for task in restarted.history()}
            for state in ("recording", "saving", "transcribing", "diarizing", "summarizing"):
                self.assertEqual(states[state], "interrupted")
            for state in ("completed", "cancelled", "failed", "interrupted"):
                self.assertEqual(states[state], state)
            self.assertEqual(restarted.recover_interrupted(), 0)

    def test_corrupt_record_does_not_hide_valid_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = TaskStore(root)
            first = store.create("first")
            second = store.create("second")
            store.update(Path(first["meeting_dir"]), message="latest", state="saving")
            for name, content in (("broken", "{"), ("wrong_type", "[]"),
                                  ("bad_state", '{"state": []}')):
                folder = root / name
                folder.mkdir()
                (folder / "task.json").write_text(content, encoding="utf-8")
            history = store.history()
            self.assertEqual([task["title"] for task in history], ["first", "second"])
            self.assertEqual(history[0]["message"], "latest")
            self.assertEqual(len(store.errors), 3)
            self.assertEqual(history[1], second)

    def test_update_write_errors_propagate_and_preserve_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary))
            task = store.create("meeting")
            with mock.patch("console_support.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    store.update(Path(task["meeting_dir"]), state="completed")
            self.assertEqual(store.history(), [task])

    def test_invalid_or_naive_timestamps_are_reported_without_hiding_valid_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary))
            valid = store.create("valid", state="completed")
            for stamp in ("not-a-date", "2026-10-16", "2026-10-16T01:00:00", ""):
                task = store.create("broken date")
                task["updated_at"] = stamp
                save_json(Path(task["meeting_dir"]) / "task.json", task)
            self.assertEqual(store.history(), [valid])
            self.assertEqual(len(store.errors), 4)
            self.assertEqual(store.recover_interrupted(), 0)

    def test_history_orders_aware_timestamps_by_instant_across_timezones(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary))
            for title, stamp in (("earlier", "2026-10-16T13:00:00+05:00"),
                                 ("later", "2026-10-16T09:00:00+00:00")):
                task = store.create(title)
                task["updated_at"] = stamp
                save_json(Path(task["meeting_dir"]) / "task.json", task)
            self.assertEqual([task["title"] for task in store.history()], ["later", "earlier"])

    def test_readonly_task_does_not_block_recovery_of_other_tasks(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = TaskStore(Path(temporary))
            writable = store.create("writable")
            readonly = store.create("readonly")
            readonly_file = Path(readonly["meeting_dir"]) / "task.json"
            original = readonly_file.read_bytes()

            def write_unless_readonly(path, data):
                if path == readonly_file:
                    raise PermissionError("task is read-only")
                save_json(path, data)

            with mock.patch("console_support.save_json", side_effect=write_unless_readonly):
                self.assertEqual(store.recover_interrupted(), 1)
            self.assertEqual(readonly_file.read_bytes(), original)
            tasks = {task["title"]: task for task in store.history()}
            self.assertEqual(tasks["readonly"], readonly)
            self.assertEqual(tasks["writable"]["state"], "interrupted")
            self.assertEqual(tasks["writable"]["meeting_dir"], writable["meeting_dir"])
            self.assertEqual(len(store.recovery_errors), 1)
            self.assertIn(str(readonly_file.parent), store.recovery_errors[0])
            self.assertIn("read-only", store.recovery_errors[0])


if __name__ == "__main__":
    unittest.main()
