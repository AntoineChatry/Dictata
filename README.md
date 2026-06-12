# Dictata

**100% local** system-wide voice dictation for Windows, in native Rust. Press a
global hotkey, speak, press again: the text is transcribed locally by
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) and pasted into the
active application. No data ever leaves the machine.

Native port of the Python version (`freewhisper`), with feature parity:
a single executable, no Python environment, GPU transcription via Vulkan.

## Features

- **Hotkey dictation** (toggle or push-to-talk), automatic paste into the
  active application, floating dock with waveform (configurable size, opacity
  and position).
- **Continuous mode (streaming)**: text is inserted as you speak, at every
  detected pause.
- **Vulkan GPU transcription** (AMD/Intel/NVIDIA) or CPU — whisper-rs with the
  `vulkan` feature.
- **Audio sources**: microphone, system audio (WASAPI loopback) or a mix of
  both (meeting mode).
- **Output modes**: raw, or post-processing through a local OpenAI-compatible
  LLM (cleanup, email, message, list…), with customizable prompts.
- **Transcribe a file** (audio/video, History page) — any format handled by
  ffmpeg.
- Custom **vocabulary and replacements** injected as the initial prompt.
- Built-in **ggml model library** (download, hardware-based recommendation).
- **fr / en / es interface**, dark theme, system tray icon, transcription
  history.

## Requirements

- Windows 10/11.
- [ffmpeg](https://ffmpeg.org/) in `PATH` (only for "Transcribe a file").
- To build with GPU support: the [Vulkan SDK](https://vulkan.lunarg.com/) and
  the Visual Studio Build Tools (CMake + Ninja included).

## Building

```powershell
cargo build --release
```

Build specifics:

- **MAX_PATH workaround (Windows)**: whisper.cpp's `vulkan-shaders-gen`
  sub-project can exceed the Windows MAX_PATH limit (260 characters) and
  break MSBuild (FTK1011/C1083). If that happens, create a local (untracked)
  `.cargo/config.toml` pointing the build to a short path at a drive root:

  ```toml
  [build]
  target-dir = "C:/wt"   # any short path
  ```

  The executable and its `config.json` then live in `<target-dir>\<profile>\`.
- After a `cargo clean`, rebuild inside a Visual Studio environment with the
  Ninja generator:

```bat
VsDevCmd.bat -arch=amd64 && set CMAKE_GENERATOR=Ninja && set CMAKE_GENERATOR_INSTANCE= && set VULKAN_SDK=<SDK path> && cargo build
```

For a CPU-only build (no Vulkan SDK required):

```powershell
cargo build --release --no-default-features
```

## Usage

1. Launch the executable: the application lives in the system tray.
2. Open **Settings** (tray menu, or set the `DICTATA_OPEN_SETTINGS=1`
   environment variable at launch): pick a model in the Models page (built-in
   download), set the hotkey, language and audio source.
3. Place the cursor where you want to write, press the hotkey (default
   `Ctrl+Alt+Space`), speak, press again. `Esc` cancels the current take.

The configuration is read from `config.json` next to the executable, or from
the directory pointed to by the `DICTATA_HOME` environment variable. The
file must be UTF-8 **without BOM**.

GPU: `gpu` config field — `"auto"` (default, uses Vulkan when available),
`"cpu"`, `"vulkan"`, `"cuda"`.

## Architecture

| Module | Role |
|---|---|
| `main.rs` | Orchestration: states, hotkey, dock, transcription threads |
| `audio.rs` | cpal capture (mic / loopback / mix), 16 kHz resampling, ffmpeg decoding |
| `transcriber.rs` | whisper-rs wrapper (model loading, transcription) |
| `streaming.rs` | Continuous mode: pause-based chunking, progressive emission |
| `settings.rs` | Settings window (8 pages, egui) — presentation |
| `settings_logic.rs` | Settings business logic, testable without UI |
| `dock.rs` | Floating dock (waveform, states) |
| `modes.rs` / `llm.rs` | Output modes and local LLM post-processing |
| `models.rs` | ggml catalog, download, paths |
| `i18n.rs` | fr/en/es translations (`tr()`) |
| `config.rs` / `history.rs` / `paste.rs` / `hotkey.rs` / `tray.rs` / `hardware.rs` / `platform.rs` | Config, history, paste, global hotkey, tray, hardware detection, Windows integration |

## Tests

```powershell
cargo test
```

21 unit tests (config, resampling, mixing, modes, settings logic…). A CLI
example is provided:

```powershell
cargo run --example transcribe -- <file.wav>   # DICTATA_GPU=1 for GPU
```

## License

Personal project — contact the author before reuse.
