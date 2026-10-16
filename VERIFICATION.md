# Windows control console verification

Verified on this Windows 11 x64 workstation, Python 3.12.8, 2026-10-16.

## Changes

- `app.py`: current meeting/environment/history console, queued main-thread UI
  dispatch, background audio finalization, persistent task state, streaming
  minutes, cancellation/close handling, single instance, rotating logs, and a
  non-interactive packaged-runtime diagnostic.
- `console_support.py`: writable user-data paths separate from bundle resources,
  atomic task/settings JSON, task history, and interrupted-run recovery.
- `environment_setup.py`, `download_models.py`: local readiness checks and model
  preparation with progress, atomic downloads, and explicit cancellation limits.
- `requirements.txt`: constrain PyAV below 19 because faster-whisper 1.2.x passes
  `metadata_errors`, which PyAV 19 removed. The delivered bundle uses PyAV 16.1.0.
- `meeting_record.spec`, `build.ps1`, `requirements-build.txt`: windowed Windows
  bundle and ZIP; explicitly include native audio/inference libraries and VAD.
- `test_console_pipeline.py`, `test_console_support.py`,
  `test_environment_setup.py`, `test_packaging.py`: regression and artifact checks.
- `README.md`, `.gitignore`, `WORK_PLAN.md`: usage, data paths, build and scope.

The existing audio capture algorithms remain intact. The GUI keeps Tkinter and
reuses the existing Recorder/Whisper/Ollama pipeline. Reprocessing writes a new
result directory rather than overwriting previous results.

## Evidence

- Original regression suite: 65 tests passed before changes to `app.py`.
- Final `python -X utf8 -m unittest discover -q`: **112 tests passed**, no skips.
- `ruff check --isolated --select E4,E7,E9,F .`: passed.
- Python compilation, `pip check`, and `git diff --check`: passed.
- Real EXE tests: Windows GUI subsystem; PortAudio relative DLL path; Whisper VAD;
  CTranslate2/Sherpa native modules; no bundled WAV/user weights; launch with an
  isolated data directory and no PYTHONHOME/PYTHONPATH, import runtime packages,
  decode 1,600 PCM samples, and initialize Tcl without opening a window.
- Real end-to-end pipeline: a 12.46-second Windows TTS Chinese meeting sample
  transcribed with local large-v3-turbo int8, processed by Sherpa, and summarized
  with local Ollama qwen2.5:7b. Completed in 111.17 seconds, no error, no fallback,
  no cancellation. Transcript contains the expected delivery date and the minutes
  contain action items. This is a short synthetic acceptance case, not a speed benchmark.
- Source GUI: the current-meeting page was visually inspected; controls and text
  panels fit, status and audio levels rendered. Layout verdict is persisted under
  `.omx/state/control-console/ralph-progress.json` (local, ignored by Git).
- Existing local Whisper cache was reused in the persistent model directory;
  optional speaker models were downloaded there. No model weights are distributed.

## Remaining limits

- The user stopped computer use with Escape. No further UI input was sent;
  environment/history pages did not receive visual acceptance testing.
- No clean Windows VM, extended hardware microphone/loopback recording, unplugging
  devices during a real meeting, or multi-hour recording was tested in this run.
- Whisper computation stops at segment boundaries; native diarization stops at
  callbacks; Ollama cancellation waits for output or the 30-second read timeout.
  The upstream Whisper downloader can only cancel after its current call returns.
- The portable EXE needs its neighboring `_internal` directory. New computers
  still need model preparation and Ollama for AI minutes; recording is independent.

Source changes are prepared on branch `feat/windows-control-console`. The Windows
bundle is generated locally; recordings and model weights are excluded from Git
and from the distributable. No binary release has been published.
