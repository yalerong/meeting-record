import struct
import tempfile
import unittest
import wave
from pathlib import Path

from app import (
    WavWriter,
    block_peak,
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


if __name__ == "__main__":
    unittest.main()
