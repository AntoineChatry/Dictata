//! Continuous dictation: transcribes and inserts text progressively while
//! recording (port of `freewhisper/streaming.py`).
//!
//! Chunks are cut on speech pauses (RMS): on each pause, the accumulated
//! audio is transcribed and emitted immediately. An emitted chunk is
//! final — the text is never rewritten (no backtracking).
//!
//! Threading invariants:
//! - `finish()` takes `self` by value, so it can only be called once
//!   (enforced by ownership); it never blocks the caller.
//! - `emit` runs synchronously on the worker thread: a slow paste delays
//!   the next chunk's transcription, but never loses audio (capture keeps
//!   accumulating in the cpal buffers and is drained on the next poll).
//! - `done` is always called exactly once, even if the worker panics
//!   (catch_unwind), so the UI can never get stuck waiting.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::audio::DrainHandle;
use crate::transcriber::Transcriber;

const SAMPLE_RATE: usize = 16000;
const POLL_MS: u64 = 200; // buffer drain cadence
const SILENCE_RMS: f32 = 0.008; // below this RMS, a 100 ms block counts as silence
const PAUSE_S: f32 = 0.7; // trailing silence that triggers a cut
const MIN_VOICED_S: f32 = 0.6; // minimum voiced audio in a chunk (anti-hallucination)
const MAX_CHUNK_S: f32 = 15.0; // forced cut even without a pause
const PROMPT_TAIL: usize = 200; // chars of already-emitted text re-injected as prompt

/// Stream transcription parameters (immutable during the session).
pub struct StreamParams {
    pub model_path: PathBuf,
    pub gpu: bool,
    pub language: Option<String>,
    pub vocab_prompt: String,
    pub beam_size: i32,
}

/// Streaming session: application-side handle.
///
/// The worker runs between `start()` and `finish()`. `emit` is called from the
/// worker for each non-empty chunk; `done` receives the full text at the end.
pub struct StreamingSession {
    stop: Arc<AtomicBool>,
    tail_tx: Sender<Vec<f32>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl StreamingSession {
    /// Starts the worker. `recorder.start()` must already have been called.
    pub fn start(
        drain: DrainHandle,
        transcriber: Arc<Mutex<Option<Transcriber>>>,
        params: StreamParams,
        emit: impl Fn(&str) + Send + 'static,
        done: impl FnOnce(Result<String, String>) + Send + 'static,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let (tail_tx, tail_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            // `done` must be called even if `run` panics, otherwise
            // the application stays stuck in the "Transcription…" state.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(drain, transcriber, params, &emit, stop2, tail_rx)
            }))
            .unwrap_or_else(|_| Err("panique du worker streaming".into()));
            done(result);
        });
        StreamingSession {
            stop,
            tail_tx,
            handle: Some(handle),
        }
    }

    /// Ends the session: `tail` is the audio returned by `recorder.stop()`
    /// (the cpal stream must be stopped by the caller, on its thread). The
    /// worker transcribes the remainder then calls `done`; this call does not
    /// block. The returned `JoinHandle` lets the caller check that the
    /// worker has fully finished before starting another one.
    #[must_use]
    pub fn finish(mut self, tail: Vec<f32>) -> std::thread::JoinHandle<()> {
        let _ = self.tail_tx.send(tail);
        self.stop.store(true, Ordering::Relaxed);
        self.handle.take().expect("finish() consomme la session, le handle est present")
    }
}

/// Trailing silence (s): 100 ms blocks below SILENCE_RMS from the end of the buffer.
fn trailing_silence(buf: &[f32]) -> f32 {
    let block = SAMPLE_RATE / 10; // 100 ms
    let mut silence = 0.0f32;
    let mut end = buf.len();
    while end >= block {
        let start = end - block;
        let rms = (buf[start..end].iter().map(|s| s * s).sum::<f32>() / block as f32).sqrt();
        if rms >= SILENCE_RMS {
            break;
        }
        silence += 0.1;
        end = start;
    }
    silence
}

fn run(
    drain: DrainHandle,
    transcriber: Arc<Mutex<Option<Transcriber>>>,
    params: StreamParams,
    emit: &(impl Fn(&str) + Send),
    stop: Arc<AtomicBool>,
    tail_rx: Receiver<Vec<f32>>,
) -> Result<String, String> {
    let mut buf: Vec<f32> = Vec::new();
    let mut text = String::new();

    let flush = |buf: &mut Vec<f32>, text: &mut String| -> Result<(), String> {
        let chunk = std::mem::take(buf);
        // Acquire the lock even if poisoned (another thread panicked):
        // an Option<Transcriber> stays coherent, at worst we reload it.
        let mut guard = transcriber.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            *guard = Some(Transcriber::load(&params.model_path, params.gpu)?);
        }
        let t = guard.as_mut().unwrap();
        // Prompt = vocabulary + tail of the already-emitted text (continuity).
        let tail: String = {
            let chars: Vec<char> = text.chars().collect();
            let skip = chars.len().saturating_sub(PROMPT_TAIL);
            chars[skip..].iter().collect()
        };
        let prompt = format!("{} {}", params.vocab_prompt, tail).trim().to_string();
        let prompt_opt = if prompt.is_empty() { None } else { Some(prompt.as_str()) };
        let piece = t.transcribe(&chunk, params.language.as_deref(), false, prompt_opt, params.beam_size, None)?;
        if !piece.is_empty() {
            let out = if text.is_empty() { piece } else { format!(" {piece}") };
            text.push_str(&out);
            emit(&out);
        }
        Ok(())
    };

    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(POLL_MS));
        buf.extend(drain.drain());
        let silence = trailing_silence(&buf);
        let voiced = buf.len() as f32 / SAMPLE_RATE as f32 - silence;
        if voiced < MIN_VOICED_S {
            continue;
        }
        if silence >= PAUSE_S || buf.len() as f32 / SAMPLE_RATE as f32 >= MAX_CHUNK_S {
            if let Err(e) = flush(&mut buf, &mut text) {
                eprintln!("[streaming] chunk: {e}");
            }
        }
    }

    // Remainder: the final audio returned by recorder.stop() on the caller side.
    if let Ok(tail) = tail_rx.recv_timeout(Duration::from_secs(5)) {
        buf.extend(tail);
    }
    let silence = trailing_silence(&buf);
    if buf.len() as f32 / SAMPLE_RATE as f32 - silence >= 0.3 {
        flush(&mut buf, &mut text)?;
    }
    Ok(text)
}
