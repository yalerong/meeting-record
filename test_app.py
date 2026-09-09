import struct
import math
import tempfile
import unittest
import wave
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

from app import (
    MAX_WAV_DATA_BYTES,
    STALL_RESTART_SEC,
    MeetingApp,
    SYSTEM_AUDIO_UNRECOGNISED_SUFFIX,
    WavWriter,
    Recorder,
    backup_candidates,
    block_peak,
    input_devices,
    is_hallucination,
    label_speakers,
    speaker_name,
    clean_markdown,
    fallback_minutes,
    level_percent,
    peak_to_dbfs,
    safe_name,
    split_text,
    timestamp,
    transcription_quality,
    is_system_audio_input,
    mix_wav_tracks,
    select_input_format,
    system_input_devices,
    input_capture_settings,
    meeting_audio_paths,
    preferred_microphone_device,
    choose_microphone_recording,
    recording_failures,
    stop_recorders_safely,
    track_start_offsets,
    verify_wav_file,
    diarization_source,
    downmix_to_mono,
)


class AppHelpersTest(unittest.TestCase):
    def test_safe_name_removes_windows_path_characters(self):
        self.assertEqual(safe_name('  周会: A/B?  '), "周会- A-B")

    def test_timestamp_formats_hours(self):
        self.assertEqual(timestamp(3661.9), "01:01:01")

    def test_split_text_preserves_all_content(self):
        text = "第一行\n第二行\n第三行"
        chunks = split_text(text, max_chars=8)
        self.assertEqual("\n".join(chunks), text)

    def test_fallback_minutes_keeps_required_sections(self):
        minutes = fallback_minutes(
            "我们决定采用方案一。小王负责明天完成。预算问题需要确认。",
            "项目会",
            Path("原始录音.wav"),
        )
        for heading in ("会议摘要", "主要决定", "行动项", "待确认问题"):
            self.assertIn(heading, minutes)

    def test_clean_markdown_removes_outer_code_fence(self):
        self.assertEqual(clean_markdown("```markdown\n# 会议纪要\n```"), "# 会议纪要")

    def test_transcription_quality_flags_low_confidence(self):
        self.assertIn("较低", transcription_quality([-0.75, -0.7]))
        self.assertIn("较好", transcription_quality([-0.2, -0.1]))

    def test_block_peak_uses_absolute_value(self):
        self.assertEqual(block_peak(struct.pack("<3h", 10, -300, 7)), 300)
        self.assertEqual(block_peak(b""), 0)

    def test_peak_to_dbfs_floor_and_full_scale(self):
        self.assertEqual(peak_to_dbfs(0), -60.0)
        self.assertAlmostEqual(peak_to_dbfs(32767), 0.0, places=2)
        self.assertEqual(level_percent(0), 0.0)
        self.assertAlmostEqual(level_percent(32767), 100.0, places=2)

    def test_is_hallucination_drops_known_subtitle_junk(self):
        self.assertTrue(is_hallucination("请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"))
        self.assertTrue(is_hallucination("字幕由志愿者提供"))
        self.assertTrue(is_hallucination("嗯。"))
        self.assertTrue(is_hallucination("   "))

    def test_is_hallucination_keeps_normal_office_speech(self):
        self.assertFalse(is_hallucination("这个邮件我转发给你了"))
        self.assertFalse(is_hallucination("大家关注一下这个数字"))
        self.assertFalse(is_hallucination("我们决定下周五交报表"))

    def test_label_speakers_picks_the_biggest_overlap(self):
        turns = [(0.0, 10.0, 0), (10.0, 20.0, 1)]
        self.assertEqual(label_speakers([(1.0, 3.0), (11.0, 12.0), (9.0, 15.0)], turns), [0, 1, 1])

    def test_label_speakers_returns_none_without_overlap(self):
        self.assertEqual(label_speakers([(30.0, 31.0)], [(0.0, 10.0, 0)]), [None])
        self.assertEqual(speaker_name(None), "未知说话人")
        self.assertEqual(speaker_name(0), "说话人1")


class WavWriterTest(unittest.TestCase):
    def test_stop_surfaces_writer_drain_timeout_without_closing_it_concurrently(self):
        recorder = Recorder(None)
        recorder._consumer = mock.Mock()
        recorder._consumer.is_alive.return_value = True
        recorder._writer = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "录音文件可能缺少末尾内容"):
            recorder.stop()

        self.assertIsNotNone(recorder.error)
        recorder._writer.close.assert_not_called()

    def test_expected_seconds_uses_wall_clock_session_length(self):
        recorder = Recorder(None)
        recorder.started_at = 100.0
        recorder.preroll_seconds = 4.0
        with mock.patch("app.time.monotonic", return_value=111.5):
            self.assertEqual(recorder.expected_seconds, 15.5)

    def test_expected_seconds_keeps_time_lost_to_device_reconnects(self):
        # A recovered stall is padded with silence, so the file must still span the gap.
        recorder = Recorder(None)
        recorder.started_at = 100.0
        recorder.stalled_seconds = 3.0
        with mock.patch("app.time.monotonic", return_value=111.5):
            self.assertEqual(recorder.expected_seconds, 11.5)

    def test_callback_pads_a_recovered_capture_gap_with_silence(self):
        recorder = Recorder(None, samplerate=10)
        block = bytearray(struct.pack("<2h", 100, 200))
        status = SimpleNamespace(input_overflow=False)
        with mock.patch("app.time.monotonic", return_value=100.2):
            recorder._callback(block, 2, None, status)
        # Next block starts 3 s after the previous one ended: a watchdog restart happened.
        with mock.patch("app.time.monotonic", return_value=103.4):
            recorder._callback(block, 2, None, status)

        queued = list(recorder._queue.queue)
        self.assertEqual(queued, [bytes(block), bytes(30 * 2), bytes(block)])
        self.assertAlmostEqual(recorder.padded_seconds, 3.0)
        self.assertIsNone(recorder.error)
        self.assertTrue(any("静音补齐" in event for event in recorder.events))

    def test_callback_does_not_pad_ordinary_scheduling_jitter(self):
        recorder = Recorder(None, samplerate=10)
        block = bytearray(struct.pack("<2h", 100, 200))
        status = SimpleNamespace(input_overflow=False)
        with mock.patch("app.time.monotonic", return_value=100.2):
            recorder._callback(block, 2, None, status)
        with mock.patch("app.time.monotonic", return_value=100.6):
            recorder._callback(block, 2, None, status)

        self.assertEqual(list(recorder._queue.queue), [bytes(block), bytes(block)])
        self.assertEqual(recorder.padded_seconds, 0.0)

    def test_preroll_handoff_inserts_the_uncaptured_gap(self):
        recorder = Recorder(None, samplerate=10, preroll=struct.pack("<10h", *range(10)))
        recorder.preroll_seconds = 1.0
        recorder.preroll_ended_at = 100.0

        with mock.patch("app.time.monotonic", return_value=101.2):
            recorder._callback(
                bytearray(struct.pack("<2h", 100, 200)),
                2,
                None,
                SimpleNamespace(input_overflow=False),
            )

        queued = list(recorder._queue.queue)
        self.assertEqual(queued, [bytes(20), struct.pack("<2h", 100, 200)])
        self.assertEqual(getattr(recorder, "audio_started_at", None), 99.0)

    def test_callback_maps_adc_capture_time_onto_the_monotonic_clock(self):
        recorder = Recorder(None, samplerate=10)

        with mock.patch("app.time.monotonic", return_value=500.0):
            recorder._callback(
                bytearray(struct.pack("<2h", 100, 200)),
                2,
                SimpleNamespace(inputBufferAdcTime=10.0, currentTime=10.5),
                SimpleNamespace(input_overflow=False),
            )

        self.assertEqual(recorder.last_data_at, 500.0)
        self.assertEqual(recorder.live_started_at, 499.5)
        self.assertEqual(getattr(recorder, "last_capture_ended_at", None), 499.7)

    def test_callback_ignores_adc_timestamps_without_a_lead(self):
        # MME and DirectSound report inputBufferAdcTime == currentTime (or both zero).
        recorder = Recorder(None, samplerate=10)
        for adc, current in ((0.0, 0.0), (10.5, 10.5)):
            recorder.live_started_at = 0.0
            with mock.patch("app.time.monotonic", return_value=500.0):
                recorder._callback(
                    bytearray(struct.pack("<2h", 100, 200)),
                    2,
                    SimpleNamespace(inputBufferAdcTime=adc, currentTime=current),
                    SimpleNamespace(input_overflow=False),
                )
            self.assertAlmostEqual(recorder.live_started_at, 499.8)

    @mock.patch("app.sd.RawInputStream")
    def test_watchdog_retries_after_a_failed_reopen(self, stream_cls):
        recorder = Recorder(Path("unused.wav"))
        recorder.last_data_at = 100.0
        stream_cls.side_effect = [RuntimeError("device gone"), mock.Mock()]

        with mock.patch("app.time.monotonic", return_value=104.0):
            recorder._restart_stream()
        self.assertIsNone(recorder._stream)
        self.assertIsNotNone(recorder.capture_error)

        # The watchdog condition must not require an open stream to try again.
        with mock.patch("app.time.monotonic", return_value=108.0):
            self.assertGreater(recorder.idle_seconds, STALL_RESTART_SEC)
            recorder._restart_stream()
        self.assertIsNotNone(recorder._stream)
        self.assertIsNone(recorder.capture_error)
        self.assertEqual(recorder.restarts, 1)

    @mock.patch("app.sd.RawInputStream")
    def test_start_failure_lets_the_consumer_finalize_the_writer(self, stream_cls):
        stream_cls.side_effect = RuntimeError("cannot open")
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "backup.wav"
        recorder = Recorder(path)

        with self.assertRaises(RuntimeError):
            recorder.start()

        self.assertIsNone(recorder._writer)
        self.assertIsNone(recorder._consumer)
        # The file is closed and can be reopened by the next backup candidate.
        with open(path, "wb"):
            pass

    @mock.patch("app.sd.RawInputStream")
    def test_stream_reconnect_keeps_write_error_visible(self, _stream):
        recorder = Recorder(Path("unused.wav"))
        recorder.last_data_at = 100.0
        recorder.write_error = "录音落盘失败：disk full"
        recorder.capture_error = "麦克风采集中断且暂时无法恢复：x"
        with mock.patch("app.time.monotonic", return_value=104.0):
            recorder._restart_stream()

        self.assertEqual(recorder.restarts, 1)
        self.assertIsNone(recorder.capture_error)
        self.assertEqual(recorder.error, "录音落盘失败：disk full")
        self.assertEqual(recorder.stalled_seconds, 4.0)

    @mock.patch("app.sd.RawInputStream")
    def test_recovered_stream_gap_is_not_a_recording_error(self, _stream):
        # The gap is padded with silence by the next callback, so processing may continue.
        recorder = Recorder(Path("unused.wav"))
        recorder.last_data_at = 100.0

        with mock.patch("app.time.monotonic", return_value=104.0):
            recorder._restart_stream()

        self.assertEqual(recorder.restarts, 1)
        self.assertIsNone(recorder.capture_error)
        self.assertIsNone(recorder.error)
        self.assertEqual(recorder.stalled_seconds, 4.0)

    def test_consumer_finalizes_writer_when_it_exits(self):
        recorder = Recorder(Path("unused.wav"))
        writer = mock.Mock()
        recorder._writer = writer
        recorder._queue.put(None)

        recorder._consume()

        writer.close.assert_called_once_with()
        self.assertIsNone(recorder._writer)
        self.assertIsNone(recorder.error)

    def test_wav_writer_refuses_to_exceed_riff_size_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            writer = WavWriter(Path(folder) / "t.wav")
            writer.frames = MAX_WAV_DATA_BYTES // 2
            with self.assertRaisesRegex(RuntimeError, "4GB"):
                writer.write(struct.pack("<2h", 1, 1))
            writer.close()

    def test_flush_leaves_a_playable_file_mid_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.wav"
            writer = WavWriter(path, samplerate=16_000, channels=1)
            writer.write(struct.pack("<8000h", *([1_000] * 8_000)))
            writer.flush()
            with wave.open(str(path)) as handle:
                self.assertEqual(handle.getnframes(), 8_000)
                self.assertEqual(handle.getframerate(), 16_000)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
            writer.write(struct.pack("<4000h", *([2_000] * 4_000)))
            writer.close()
            with wave.open(str(path)) as handle:
                self.assertEqual(handle.getnframes(), 12_000)
                self.assertEqual(len(handle.readframes(12_000)), 24_000)

    @mock.patch("os.fsync")
    def test_flush_syncs_audio_to_storage(self, fsync):
        with tempfile.TemporaryDirectory() as folder:
            writer = WavWriter(Path(folder) / "t.wav")
            writer.write(struct.pack("<100h", *([1_000] * 100)))
            writer.flush()
            writer.close()
        self.assertGreaterEqual(fsync.call_count, 1)

    def test_recorder_closes_writer_and_surfaces_write_failure(self):
        recorder = Recorder(Path("unused.wav"))
        writer = mock.Mock()
        writer.write.side_effect = OSError("disk full")
        recorder._writer = writer
        recorder._queue.put(struct.pack("<100h", *([1_000] * 100)))
        recorder._queue.put(None)

        recorder._consume()

        writer.close.assert_called_once_with()
        self.assertIsNone(recorder._writer)
        error = recorder.error
        assert error is not None
        self.assertIn("disk full", error)


class AudioSourceTest(unittest.TestCase):
    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_prefers_wasapi_variant_of_the_windows_default_microphone(
        self, query_devices, query_hostapis
    ):
        devices = [
            {"name": "Microphone Array", "max_input_channels": 2, "hostapi": 0},
            {"name": "Microphone Array", "max_input_channels": 2, "hostapi": 1},
            {"name": "USB Microphone", "max_input_channels": 1, "hostapi": 0},
        ]
        query_devices.return_value = devices
        query_hostapis.side_effect = lambda index: {
            "name": ("MME", "Windows WASAPI")[index]
        }

        selected = preferred_microphone_device(0, [0, 1, 2])

        self.assertEqual(selected, 1)

    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_missing_windows_default_microphone_falls_back_to_first_input(self, query_devices, _hostapis):
        query_devices.return_value = [
            {"name": "Microphone Array", "max_input_channels": 2, "hostapi": 0},
            {"name": "USB Microphone", "max_input_channels": 1, "hostapi": 0},
        ]

        self.assertEqual(preferred_microphone_device(-1, [0, 1]), 0)

    @mock.patch("app.sd.WasapiSettings")
    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_microphone_backup_candidates_exclude_system_audio_inputs(
        self, query_devices, query_hostapis, _wasapi_settings
    ):
        devices = [
            {"name": "Microphone", "hostapi": 0, "max_input_channels": 1},
            {"name": "Stereo Mix", "hostapi": 1, "max_input_channels": 2},
            {"name": "USB Microphone", "hostapi": 1, "max_input_channels": 1},
        ]
        query_devices.side_effect = lambda index=None: devices if index is None else devices[index]
        query_hostapis.side_effect = lambda index=None: (
            [{"name": "MME"}, {"name": "Windows WASAPI"}]
            if index is None
            else [{"name": "MME"}, {"name": "Windows WASAPI"}][index]
        )
        with mock.patch.object(type(__import__("app").sd.default), "device", (0, 0)):
            candidates = backup_candidates(0)

        self.assertEqual([index for index, _extra in candidates], [2])

    @mock.patch("app.sd.WasapiSettings")
    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_microphone_backup_candidates_exclude_the_selected_system_endpoint(
        self, query_devices, query_hostapis, _wasapi_settings
    ):
        # A virtual cable carries system audio under a name the hint list does not know.
        devices = [
            {"name": "Microphone", "hostapi": 0, "max_input_channels": 1},
            {"name": "CABLE Output", "hostapi": 0, "max_input_channels": 2},
            {"name": "CABLE Output", "hostapi": 1, "max_input_channels": 2},
            {"name": "USB Microphone", "hostapi": 1, "max_input_channels": 1},
        ]
        query_devices.side_effect = lambda index=None: devices if index is None else devices[index]
        query_hostapis.side_effect = lambda index=None: (
            [{"name": "MME"}, {"name": "Windows WASAPI"}]
            if index is None
            else [{"name": "MME"}, {"name": "Windows WASAPI"}][index]
        )
        with mock.patch.object(type(__import__("app").sd.default), "device", (0, 0)):
            candidates = backup_candidates(0, exclude=2)

        self.assertEqual([index for index, _extra in candidates], [3])

    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_microphone_device_list_excludes_system_audio_inputs(self, query_devices, query_hostapis):
        query_devices.return_value = [
            {"name": "麦克风阵列", "max_input_channels": 2, "hostapi": 0},
            {"name": "立体声混音", "max_input_channels": 2, "hostapi": 1},
            {"name": "电脑扬声器", "max_input_channels": 1, "hostapi": 1},
        ]
        query_hostapis.side_effect = lambda index: {"name": ("MME", "Windows WDM-KS")[index]}

        self.assertEqual(input_devices(), [(0, "麦克风阵列  [MME]")])

    def test_identifies_common_windows_system_audio_capture_names(self):
        for name in (
            "立体声混音 (Realtek HD Audio Stereo input)",
            "电脑扬声器 (Realtek HD Audio output with SST)",
            "Stereo Mix (Realtek Audio)",
            "What U Hear (Sound Blaster)",
            "Speakers [Loopback]",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_system_audio_input(name))

    def test_does_not_treat_microphones_as_system_audio(self):
        for name in ("麦克风阵列 (Realtek Audio)", "USB Conference Microphone"):
            with self.subTest(name=name):
                self.assertFalse(is_system_audio_input(name))

    def test_selects_first_supported_format_in_preference_order(self):
        attempts = []

        def supports(rate, channels):
            attempts.append((rate, channels))
            return (rate, channels) == (48_000, 2)

        self.assertEqual(select_input_format(48_000, 2, supports), (48_000, 2))
        self.assertEqual(attempts, [(16_000, 1), (48_000, 1), (48_000, 2)])

    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_lists_system_audio_endpoints_first_and_marks_other_inputs(self, query_devices, query_hostapis):
        query_devices.return_value = [
            {"name": "CABLE Output (VB-Audio Virtual Cable)", "max_input_channels": 2, "hostapi": 0},
            {"name": "立体声混音", "max_input_channels": 2, "hostapi": 1},
            {"name": "扬声器", "max_input_channels": 0, "hostapi": 1},
        ]
        query_hostapis.side_effect = lambda index: {"name": ("MME", "Windows WDM-KS")[index]}

        self.assertEqual(
            system_input_devices(),
            [
                (1, "立体声混音  [Windows WDM-KS]"),
                (0, "CABLE Output (VB-Audio Virtual Cable)  [MME]" + SYSTEM_AUDIO_UNRECOGNISED_SUFFIX),
            ],
        )

    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_system_audio_candidates_prefer_primary_speaker_endpoint(
        self, query_devices, query_hostapis
    ):
        query_devices.return_value = [
            {"name": "立体声混音 (Realtek HD Audio Stereo input)", "max_input_channels": 2, "hostapi": 0},
            {
                "name": "电脑扬声器 (Realtek HD Audio 2nd output with SST)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
            {
                "name": "电脑扬声器 (Realtek HD Audio output with SST)",
                "max_input_channels": 2,
                "hostapi": 0,
            },
        ]
        query_hostapis.return_value = {"name": "Windows WDM-KS"}

        self.assertEqual([index for index, _name in system_input_devices()], [2, 0, 1])

    def test_recorder_duration_uses_its_actual_sample_rate(self):
        recorder = Recorder(None, samplerate=48_000, channels=2)
        recorder.frames_written = 24_000
        self.assertEqual(recorder.recorded_seconds, 0.5)

    def test_stereo_capture_is_downmixed_to_mono_before_queueing(self):
        self.assertEqual(downmix_to_mono(struct.pack("<4h", 100, 300, -200, 200), 2), struct.pack("<2h", 200, 0))
        self.assertEqual(downmix_to_mono(b"ab", 1), b"ab")
        recorder = Recorder(None, samplerate=48_000, channels=2)
        recorder._callback(bytearray(struct.pack("<4h", 100, 300, -200, 200)), 2, None, SimpleNamespace(input_overflow=False))
        self.assertEqual(recorder._queue.get_nowait(), struct.pack("<2h", 200, 0))

    @mock.patch("app.sd.WasapiSettings")
    @mock.patch("app.sd.check_input_settings")
    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_wasapi_capture_enables_format_conversion(
        self, query_devices, query_hostapis, check_input_settings, wasapi_settings
    ):
        query_devices.return_value = {
            "name": "麦克风阵列",
            "max_input_channels": 2,
            "hostapi": 2,
            "default_samplerate": 48_000,
        }
        query_hostapis.return_value = {"name": "Windows WASAPI"}
        sentinel = object()
        wasapi_settings.return_value = sentinel

        rate, channels, extra = input_capture_settings(9)

        self.assertEqual((rate, channels, extra), (16_000, 1, sentinel))
        wasapi_settings.assert_called_once_with(auto_convert=True)
        check_input_settings.assert_called_once_with(
            device=9,
            samplerate=16_000,
            channels=1,
            dtype="int16",
            extra_settings=sentinel,
        )

    @mock.patch("app.sd.check_input_settings")
    @mock.patch("app.sd.query_hostapis")
    @mock.patch("app.sd.query_devices")
    def test_capture_falls_back_to_native_stereo(self, query_devices, query_hostapis, check_input_settings):
        query_devices.return_value = {
            "name": "立体声混音",
            "max_input_channels": 2,
            "hostapi": 3,
            "default_samplerate": 48_000,
        }
        query_hostapis.return_value = {"name": "Windows WDM-KS"}
        check_input_settings.side_effect = [RuntimeError("bad rate"), RuntimeError("bad channels"), None]

        self.assertEqual(input_capture_settings(12), (48_000, 2, None))


class AudioMixTest(unittest.TestCase):
    @staticmethod
    def _write_pcm(path: Path, rate: int, channels: int, frames: Sequence[tuple[int, ...]]) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            flattened = [sample for frame in frames for sample in frame]
            handle.writeframes(struct.pack(f"<{len(flattened)}h", *flattened))

    def test_mixes_mic_and_system_tracks_into_playable_mono_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mic = root / "mic.wav"
            system = root / "system.wav"
            mixed = root / "mixed.wav"
            self._write_pcm(mic, 8_000, 1, [(1_000,)] * 8_000)
            self._write_pcm(system, 16_000, 2, [(2_000, 2_000)] * 16_000)

            mix_wav_tracks([mic, system], mixed, output_rate=16_000)

            with wave.open(str(mixed)) as handle:
                self.assertEqual(handle.getframerate(), 16_000)
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getnframes(), 16_000)
                samples = struct.unpack("<16000h", handle.readframes(16_000))
            self.assertTrue(all(2_990 <= sample <= 3_010 for sample in samples[10:-10]))

    def test_mix_of_two_live_voices_has_no_per_sample_gain_switching(self):
        rate = 16_000
        mic = [(int(6_000 * math.sin(2 * math.pi * 220 * i / rate)),) for i in range(rate)]
        system = [(int(6_000 * math.sin(2 * math.pi * 330 * i / rate)),) for i in range(rate)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_pcm(root / "mic.wav", rate, 1, mic)
            self._write_pcm(root / "system.wav", rate, 1, system)

            mix_wav_tracks([root / "mic.wav", root / "system.wav"], root / "mixed.wav")

            with wave.open(str(root / "mixed.wav")) as handle:
                mixed = struct.unpack(f"<{rate}h", handle.readframes(rate))
        worst = max(abs(m - (a[0] + b[0])) for m, a, b in zip(mixed, mic, system, strict=True))
        self.assertLessEqual(worst, 1)

    def test_mix_clips_instead_of_wrapping_when_both_sides_are_loud(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self._write_pcm(root / "a.wav", 8_000, 1, [(20_000,)] * 800)
            self._write_pcm(root / "b.wav", 8_000, 1, [(20_000,)] * 800)

            mix_wav_tracks([root / "a.wav", root / "b.wav"], root / "mixed.wav", output_rate=8_000)

            with wave.open(str(root / "mixed.wav")) as handle:
                self.assertEqual(set(struct.unpack("<800h", handle.readframes(800))), {32_767})

    def test_mix_does_not_quiet_the_only_active_track(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            active = root / "active.wav"
            silent = root / "silent.wav"
            mixed = root / "mixed.wav"
            self._write_pcm(active, 16_000, 1, [(1_000,)] * 16_000)
            self._write_pcm(silent, 16_000, 1, [(0,)] * 16_000)

            mix_wav_tracks([active, silent], mixed)

            with wave.open(str(mixed)) as handle:
                samples = struct.unpack("<16000h", handle.readframes(16_000))
            self.assertTrue(all(990 <= sample <= 1_010 for sample in samples))

    def test_mix_offsets_tracks_to_align_different_preroll_lengths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            long_preroll = root / "long-preroll.wav"
            short_preroll = root / "short-preroll.wav"
            mixed = root / "mixed.wav"
            self._write_pcm(
                long_preroll,
                8_000,
                1,
                [(0,)] * 8_000 + [(1_000,)] * 8_000,
            )
            self._write_pcm(short_preroll, 8_000, 1, [(2_000,)] * 8_000)

            mix_wav_tracks(
                [long_preroll, short_preroll],
                mixed,
                output_rate=8_000,
                start_offsets=[0.0, 1.0],
            )

            with wave.open(str(mixed)) as handle:
                first = struct.unpack("<100h", handle.readframes(100))
                handle.setpos(8_100)
                aligned = struct.unpack("<100h", handle.readframes(100))
            self.assertTrue(all(sample == 0 for sample in first))
            self.assertTrue(all(sample == 3_000 for sample in aligned))

    def test_mix_keeps_longer_track_instead_of_truncating_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            short = root / "short.wav"
            long = root / "long.wav"
            mixed = root / "mixed.wav"
            self._write_pcm(short, 8_000, 1, [(1_000,)] * 4_000)
            self._write_pcm(long, 8_000, 1, [(2_000,)] * 8_000)

            mix_wav_tracks([short, long], mixed, output_rate=8_000)

            with wave.open(str(mixed)) as handle:
                self.assertEqual(handle.getnframes(), 8_000)
                handle.setpos(6_000)
                tail = struct.unpack("<100h", handle.readframes(100))
            self.assertTrue(all(sample == 2_000 for sample in tail))

    def test_verifies_playable_wav_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            self._write_pcm(path, 8_000, 1, [(1_000,)] * 8_000)
            self.assertEqual(verify_wav_file(path, expected_seconds=1.0), 1.0)

    def test_diarization_source_resamples_non_16k_mono_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / "native.wav"
            ready = root / "ready.wav"
            copy = root / "copy.wav"
            self._write_pcm(native, 48_000, 2, [(1_000, 3_000)] * 48_000)
            self._write_pcm(ready, 16_000, 1, [(1_000,)] * 16_000)

            self.assertIs(diarization_source(ready, copy), ready)
            self.assertFalse(copy.exists())
            self.assertEqual(diarization_source(native, copy), copy)
            with wave.open(str(copy)) as handle:
                self.assertEqual((handle.getframerate(), handle.getnchannels(), handle.getnframes()), (16_000, 1, 16_000))
                samples = struct.unpack("<16000h", handle.readframes(16_000))
            self.assertTrue(all(1_990 <= sample <= 2_010 for sample in samples[10:-10]))

    def test_downsampling_removes_content_above_the_output_nyquist(self):
        # 48 kHz source with a 1 kHz tone (must survive) plus a 20 kHz tone (must not
        # alias back into the 16 kHz output as a 4 kHz tone).
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            native = root / "native.wav"
            mixed = root / "mixed.wav"
            frames = [
                (int(8_000 * math.sin(2 * math.pi * 1_000 * i / 48_000) + 8_000 * math.sin(2 * math.pi * 20_000 * i / 48_000)),)
                for i in range(48_000)
            ]
            self._write_pcm(native, 48_000, 1, frames)
            mix_wav_tracks([native], mixed)
            with wave.open(str(mixed)) as handle:
                self.assertEqual(handle.getframerate(), 16_000)
                samples = struct.unpack("<16000h", handle.readframes(16_000))
            body = samples[200:-200]
            n = len(body)
            def amplitude(freq: int) -> float:
                re = sum(v * math.cos(2 * math.pi * freq * i / 16_000) for i, v in enumerate(body))
                im = sum(v * math.sin(2 * math.pi * freq * i / 16_000) for i, v in enumerate(body))
                return 2 * math.hypot(re, im) / n
            self.assertGreater(amplitude(1_000), 7_000)
            self.assertLess(amplitude(4_000), 400)

    def test_rejects_audio_that_is_much_shorter_than_recording_session(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "truncated.wav"
            self._write_pcm(path, 8_000, 1, [(1_000,)] * 8_000)
            with self.assertRaisesRegex(RuntimeError, "时长异常"):
                verify_wav_file(path, expected_seconds=10.0)


class RecordingPlanTest(unittest.TestCase):
    def test_track_offsets_use_actual_audio_origins(self):
        microphone = cast(
            Recorder,
            SimpleNamespace(audio_started_at=70.0, started_at=100.0, preroll_seconds=30.0),
        )
        system = cast(
            Recorder,
            SimpleNamespace(audio_started_at=70.2, started_at=100.8, preroll_seconds=30.0),
        )
        for actual, expected in zip(track_start_offsets(microphone, system), [0.0, 0.2], strict=True):
            self.assertAlmostEqual(actual, expected)

        short_preroll = cast(
            Recorder,
            SimpleNamespace(audio_started_at=96.4, started_at=100.5, preroll_seconds=4.0),
        )
        for actual, expected in zip(track_start_offsets(microphone, short_preroll), [0.0, 26.4], strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_take_preroll_keeps_the_last_capture_boundary(self):
        app = object.__new__(MeetingApp)
        monitor = mock.Mock(last_capture_ended_at=123.4, last_data_at=999.0)
        monitor.preroll_bytes.return_value = b"audio"
        app.monitor = monitor
        app.system_monitor = None

        self.assertEqual(app._take_preroll(), (b"audio", 123.4))
        monitor.stop.assert_called_once_with()

    def test_complete_mix_rejects_a_track_with_a_recovered_gap(self):
        app = object.__new__(MeetingApp)
        app.audio_path = Path("complete.wav")
        microphone_path = Path("mic.wav")
        microphone = cast(
            Recorder,
            SimpleNamespace(
                label="麦克风",
                error="录音采集曾中断，时间轴丢失约 3.0 秒",
                path=microphone_path,
                audio_started_at=100.0,
                recorded_seconds=60.0,
            ),
        )
        system = cast(
            Recorder,
            SimpleNamespace(
                label="系统声音",
                error=None,
                path=Path("system.wav"),
                audio_started_at=100.0,
                recorded_seconds=60.0,
            ),
        )

        with (
            mock.patch("app.mix_wav_tracks") as mix,
            mock.patch("app.verify_wav_file"),
            self.assertRaisesRegex(RuntimeError, "时间轴不完整"),
        ):
            app._mix_complete_recording(microphone, microphone_path, system)
        mix.assert_not_called()

    def test_stopping_recorders_attempts_every_track_after_one_fails(self):
        primary = mock.Mock(label="麦克风")
        primary.stop.side_effect = OSError("disk full")
        system = mock.Mock(label="系统声音")

        errors = stop_recorders_safely(primary, None, system)

        system.stop.assert_called_once_with()
        self.assertEqual(errors, ["麦克风：disk full"])

    def test_online_mode_preserves_sources_and_has_a_complete_output(self):
        root = Path("meeting")
        self.assertEqual(
            meeting_audio_paths(root, online=True),
            {
                "primary": root / "麦克风录音.wav",
                "backup": root / "麦克风备份.wav",
                "system": root / "系统声音.wav",
                "complete": root / "完整录音.wav",
            },
        )

    def test_offline_mode_keeps_original_file_names(self):
        root = Path("meeting")
        self.assertEqual(
            meeting_audio_paths(root, online=False),
            {
                "primary": root / "原始录音.wav",
                "backup": root / "备份录音.wav",
                "system": None,
                "complete": root / "原始录音.wav",
            },
        )

    def test_chooses_audible_complete_backup_when_primary_stalls(self):
        primary = cast(Recorder, SimpleNamespace(
            path=Path("primary.wav"), max_peak=2_000, recorded_seconds=5.0, error=None
        ))
        backup = cast(Recorder, SimpleNamespace(
            path=Path("backup.wav"), max_peak=1_000, recorded_seconds=60.0, error=None
        ))
        self.assertEqual(choose_microphone_recording(primary, backup), backup.path)

    def test_keeps_primary_when_backup_is_silent(self):
        primary = cast(Recorder, SimpleNamespace(
            path=Path("primary.wav"), max_peak=2_000, recorded_seconds=60.0, error=None
        ))
        backup = cast(Recorder, SimpleNamespace(
            path=Path("backup.wav"), max_peak=0, recorded_seconds=61.0, error=None
        ))
        self.assertEqual(choose_microphone_recording(primary, backup), primary.path)

    def test_collects_write_failures_for_prominent_user_warning(self):
        good = cast(Recorder, SimpleNamespace(label="麦克风", error=None))
        bad = cast(Recorder, SimpleNamespace(label="系统声音", error="录音落盘失败：disk full"))
        self.assertEqual(recording_failures(good, None, bad), ["系统声音：录音落盘失败：disk full"])


if __name__ == "__main__":
    unittest.main()
