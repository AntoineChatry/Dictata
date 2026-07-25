# Changelog

All notable changes to Dictata are documented here.
Format loosely based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — 2026-07-25

First public release. **Windows 10/11 only** — see [LINUX.md](LINUX.md) for the
state of Linux support.

### Features

- **Hotkey dictation** (toggle or push-to-talk), automatic paste into the
  active application, floating dock with waveform (size, opacity and position
  configurable).
- **Continuous mode (streaming)**: text inserted as you speak, at every
  detected pause.
- **Silence skipping (VAD)** on standard dictation, via a small (~2 MB)
  whisper.cpp model downloaded on first use.
- **Vulkan GPU transcription** (AMD/Intel/NVIDIA) or CPU.
- **Audio sources**: microphone, system audio (WASAPI loopback), or both mixed.
- **Output modes**: raw, or post-processing through a local OpenAI-compatible
  LLM, with customisable prompts.
- **Automatic mode per application**: rules matching the foreground executable
  or window title.
- **Transcribe a file** (audio/video, via ffmpeg).
- Custom **vocabulary and replacements** injected as the initial prompt.
- Built-in **ggml model library** with hardware-aware recommendations, plus
  **HuggingFace search** for any ggml `.bin` model.
- **fr / en / es interface**, dark theme, system tray, transcription history.
- Executable metadata: the binary carries its own icon, version, product name
  and copyright, so Explorer, the taskbar and Task Manager show Dictata rather
  than a nameless generic application. The binary is **not** code-signed —
  Windows SmartScreen will still warn on first run.

### Behaviour worth knowing

- **Escape must be held (~400 ms) to cancel a take.** Escape is read globally,
  so a plain tap — the one meant for the focused application, closing a popup
  or a dialog — used to abort the dictation. A deliberate hold is now required.
- **History is enabled by default**, capped at 500 entries; both are
  configurable in Settings. Entries are stored in clear text in
  `history.jsonl` next to the executable.
- **Local by default**: nothing is sent anywhere. The LLM endpoint is a free
  text field, so Settings now warns when it points somewhere other than this
  machine — pointing it at a remote host sends the full text of every
  dictation there.
- **Crash recovery**: a malformed model file makes whisper.cpp `abort()`, which
  no error handling can intercept. A marker file survives the abort so the next
  launch falls back to the default model instead of crashing again, and says so.
- **Configuration is forward-compatible**: missing keys fall back to defaults,
  so a `config.json` from an earlier build keeps every setting it had.

### Known limitations

- Windows only (Linux: see [LINUX.md](LINUX.md)).
- `<think>` blocks from reasoning LLMs are not stripped — use a non-reasoning
  model for post-processing.
- Downloaded models are not checksum-verified.
- The history file is not encrypted.
- "Transcribe a file" requires ffmpeg in `PATH`.
