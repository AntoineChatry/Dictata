# Dictata on Linux — status

**Short answer: Linux is not supported in v0.1.0.** The code compiles for Linux
in principle (every Windows-only call has a `#[cfg(not(windows))]` fallback),
but several of those fallbacks are no-ops that disable the core of the
application. No Linux build has ever been produced or run: the project is
developed on Windows, and cross-compiling whisper.cpp from Windows would not
prove anything anyway.

`scripts/release-linux.sh` exists so the packaging step is ready when the
blockers below are lifted. It is **unverified** — treat it as a starting point,
not as a working build.

## What already works on Linux (by design)

| Piece | Why it is fine |
|---|---|
| Transcription (`whisper-rs`) | whisper.cpp is cross-platform; CPU build needs no SDK |
| Audio capture (`cpal`) | ALSA backend, microphone input |
| UI (`eframe`/`egui`) | cross-platform |
| Clipboard (`arboard`) | X11 supported |
| Config, history, models, modes, i18n | pure Rust, no platform calls |

## Blockers — each needs code, not packaging

1. **The dock steals focus.** `platform::make_no_activate` applies
   `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW` on Windows and is a **no-op**
   elsewhere (`src/platform.rs:28`). On Linux the dock window would take the
   focus, so the simulated `Ctrl+V` would paste into the dock instead of the
   target application. This breaks the primary function of the app. An X11
   equivalent (`_NET_WM_WINDOW_TYPE_DOCK` / override-redirect) has to be
   written and tested.

2. **The tray needs a GTK event loop.** `tray-icon` requires the tray to live
   on a thread running an **active GTK event loop** (upstream docs,
   "Thread and Event Loop Requirements"). Dictata creates the tray from the
   eframe/winit thread (`src/tray.rs`), which runs a winit loop, not a GTK one.
   Linux needs a dedicated GTK thread, with the mode/settings/quit actions
   forwarded across it.

3. **Wayland is out.** `global-hotkey` supports **Linux X11 only** (other UNIX
   targets get a no-op implementation), and `enigo` defaults to X11 (`x11rb`),
   its Wayland/libei backends being experimental and behind feature flags.
   On a Wayland session — the default on current GNOME, Ubuntu 22.04+ and
   Fedora — neither the global hotkey nor the paste would work.

4. **Cancelling a take does nothing.** `platform::escape_down` returns `false`
   outside Windows (`src/platform.rs:106`), so the long-press-Escape cancel
   gesture can never fire.

5. **No automatic mode per application.** `platform::foreground_app` returns
   `None` outside Windows (`src/platform.rs:87`): the app-to-mode rules are
   inert and the manually selected mode is kept.

6. **No system-audio capture.** The "system audio" and "mix" sources rely on
   WASAPI loopback, which is Windows-only in `cpal` — an input stream opened
   on the default *output* device (`src/audio.rs:217`). The
   Linux equivalent is a PulseAudio/PipeWire *monitor* source; whether it shows
   up as a `cpal` input device has not been checked.

7. **RAM/GPU detection is degraded.** `hardware.rs` has non-Windows fallbacks,
   so the model recommendations would be based on incomplete information.

## Build dependencies (Debian/Ubuntu)

```bash
sudo apt install build-essential cmake pkg-config \
    libasound2-dev libgtk-3-dev libxdo-dev libx11-dev \
    libayatana-appindicator3-dev
```

The Vulkan build additionally needs the Vulkan SDK (`libvulkan-dev` and a
shader compiler). The CPU build — `cargo build --release --no-default-features`
— needs none of that and is the sane first target.

## Suggested order of work

1. Get the CPU build to compile and run on X11 (blockers 2 then 1).
2. Restore the cancel gesture (4) — it is a small X11 key-state query.
3. Decide on Wayland (3): it changes the whole input strategy
   (XDG portals / libei) and is the largest piece of work by far.
4. Everything else is a nice-to-have.
