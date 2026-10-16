"""Inspect the real Windows bundle: python -m unittest test_packaging -v."""
from pathlib import Path
import json
import os
import struct
import subprocess
import tempfile
import unittest


BUNDLE = Path(__file__).resolve().parent / "dist" / "MeetingRecord"
EXE = BUNDLE / "MeetingRecord.exe"


@unittest.skipUnless(EXE.is_file(), "Build the Windows bundle first with build.ps1")
class WindowsBundleTests(unittest.TestCase):
    def test_packaged_runtime_decodes_audio_without_system_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "diagnostics.json"
            environment = os.environ.copy()
            environment["MEETING_RECORD_HOME"] = str(root / "persistent-data")
            environment.pop("PYTHONHOME", None)
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [str(EXE), "--diagnostics-file", str(output)], env=environment,
                cwd=root, timeout=45, capture_output=True,
            )
            self.assertEqual(result.returncode, 0)
            self.assertTrue(output.is_file(), "Packaged startup did not write diagnostics")
            diagnostics = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(diagnostics["frozen"])
            self.assertEqual(Path(diagnostics["data_dir"]), root / "persistent-data")
            self.assertFalse(any(value.startswith("ERROR:") for value in diagnostics["modules"].values()))
            self.assertNotIn("runtime_error", diagnostics)
            self.assertEqual(diagnostics["audio_decode_frames"], 1600)
            self.assertTrue(diagnostics["tcl_version"].startswith("8."))

    def test_executable_uses_windows_gui_subsystem(self):
        # A console-subsystem executable opens an unwanted terminal on double click.
        with EXE.open("rb") as executable:
            self.assertEqual(executable.read(2), b"MZ")
            executable.seek(0x3C)
            pe_offset = struct.unpack("<I", executable.read(4))[0]
            executable.seek(pe_offset)
            self.assertEqual(executable.read(4), b"PE\0\0")
            executable.seek(pe_offset + 24 + 68)
            self.assertEqual(struct.unpack("<H", executable.read(2))[0], 2)

    def test_portaudio_keeps_runtime_lookup_directory(self):
        directory = BUNDLE / "_internal" / "_sounddevice_data" / "portaudio-binaries"
        self.assertTrue(list(directory.glob("libportaudio*.dll")), "PortAudio DLL missing")

    def test_transcription_and_diarization_native_modules_are_bundled(self):
        internal = BUNDLE / "_internal"
        self.assertTrue(list((internal / "ctranslate2").rglob("_ext*.pyd")))
        self.assertTrue(list((internal / "sherpa_onnx").rglob("*sherpa*.pyd")))
        sherpa_libraries = internal / "sherpa_onnx" / "lib"
        self.assertTrue((sherpa_libraries / "onnxruntime.dll").is_file())
        self.assertTrue((sherpa_libraries / "sherpa-onnx-cxx-api.dll").is_file())

    def test_voice_activity_detection_asset_is_bundled(self):
        assets = BUNDLE / "_internal" / "faster_whisper" / "assets"
        self.assertTrue(list(assets.glob("*.onnx")), "faster-whisper VAD model missing")

    def test_distribution_has_no_developer_recordings_or_models(self):
        for directory in (BUNDLE, BUNDLE / "_internal"):
            for name in ("recordings", "models", ".env"):
                self.assertFalse((directory / name).exists(), f"Private build input: {name}")
        self.assertFalse(list(BUNDLE.rglob("*.wav")), "A recording was accidentally bundled")
        self.assertFalse(list(BUNDLE.rglob("model.bin")), "A user transcription model was bundled")


if __name__ == "__main__":
    unittest.main()
