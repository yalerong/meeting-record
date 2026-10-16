# Windows control console work plan

Scope: keep the existing Tkinter UI and recording implementation. Deliver a Windows
executable, persistent user data, visible environment/task/history status, and
responsive save/cancel operations. Do not change audio capture algorithms.

1. Run the existing regression suite before editing `app.py`.
2. Add regression tests for cancellation beyond transcription, streaming output,
   UI event dispatch, and background recording finalization before changing those paths.
3. Separate bundled resources from writable user data; persist task state atomically.
4. Add console tabs for the current task, environment preparation, and history.
   Keep all Tk calls on the main thread using a queue consumed by `after`.
5. Move recording finalization to a background thread; propagate cancellation
   through model loading boundaries, diarization callbacks, and Ollama streaming.
   Show bounded waits honestly; never claim a cancelled result completed.
6. Build a windowed PyInstaller application and distributable ZIP. Keep model
   weights outside the bundle and preserve old recordings without moving them.
7. Verify tests, lint, compilation, source UI, and packaged startup. Inspect the
   console visually and record evidence. Do not claim clean-machine, GPU, or
   long-meeting hardware validation unless actually exercised.

Acceptance: launching the EXE displays a responsive console without Python setup;
recordings survive restarts; environment readiness is visible; task failures and
interruption remain inspectable; cancellation does not publish a completed task.
