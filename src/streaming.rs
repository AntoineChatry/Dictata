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
// Lower bar used when there is no choice but to decide now: the end of the take,
// and the forced cut at MAX_CHUNK_S. Below it, the audio is treated as noise.
const MIN_TAIL_VOICED_S: f32 = 0.3;
const MAX_CHUNK_S: f32 = 15.0; // forced cut even without a pause
const PROMPT_TAIL: usize = 200; // chars of already-emitted text re-injected as prompt

/// Stream transcription parameters (immutable during the session).
pub struct StreamParams {
    pub model_path: PathBuf,
    pub gpu: bool,
    pub language: Option<String>,
    pub vocab_prompt: String,
    pub beam_size: i32,
    /// Amplify quiet chunks before transcription (`low_voice` in the config).
    /// The one-shot path has always honoured this setting; streaming did not,
    /// which made it noticeably worse on a soft microphone.
    pub low_voice: bool,
}

/// Streaming session: application-side handle.
///
/// The worker runs between `start()` and `finish()`. `emit` is called from the
/// worker for each non-empty chunk; `done` receives the full text at the end.
pub struct StreamingSession {
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
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
        let cancel = Arc::new(AtomicBool::new(false));
        let cancel2 = cancel.clone();
        let (tail_tx, tail_rx) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            // `done` must be called even if `run` panics, otherwise
            // the application stays stuck in the "Transcription…" state.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(drain, transcriber, params, &emit, stop2, cancel2, tail_rx)
            }))
            .unwrap_or_else(|_| Err("streaming worker panicked".into()));
            done(result);
        });
        StreamingSession {
            stop,
            cancel,
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
        self.handle.take().expect("finish() consumes the session, the handle is present")
    }

    /// Aborts the session: the worker stops without transcribing or emitting
    /// the remaining audio. Chunks already emitted have already been pasted
    /// into the target application and cannot be taken back.
    #[must_use]
    pub fn cancel(mut self) -> std::thread::JoinHandle<()> {
        self.cancel.store(true, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.tail_tx.send(Vec::new());
        self.handle.take().expect("cancel() consumes the session, the handle is present")
    }
}

const BLOCK_SAMPLES: usize = SAMPLE_RATE / 10; // 100 ms
const BLOCK_S: f32 = 0.1;

/// Rolling measure of a streaming buffer: how much voiced audio it holds and
/// how much silence currently trails it.
///
/// Fed only with the samples newly drained from the capture, so the cost of a
/// poll is proportional to the new audio and not to the buffer already
/// accumulated. The previous implementation rescanned the whole buffer every
/// 200 ms and never stopped early on an all-silent one, which made a long pause
/// quadratic: five minutes of silence meant re-reading 4.8 M samples five times
/// a second.
///
/// [`SpeechMeter::voiced_s`] measures the *span* of speech — from the first
/// block above `SILENCE_RMS` to the last one — not the sum of the loud blocks.
///
/// This distinction caused a regression worth recording. Summing only the loud
/// blocks looks stricter and more honest, but `MIN_VOICED_S` was calibrated
/// against the span: under the old `duration - trailing_silence` formula a
/// single loud block ending 0.6 s into the chunk cleared the gate, whereas
/// summing demands six of them. Ordinary speech is full of sub-threshold gaps
/// between words, so short utterances stopped triggering a flush entirely and
/// the take came back empty. Measuring the span keeps the constant meaning what
/// it meant when it was tuned, while still reporting exactly `0.0` on a buffer
/// that never crossed the threshold — the property the incremental rewrite was
/// introduced to guarantee. Leading silence, which the old formula wrongly
/// counted as speech, is excluded.
struct SpeechMeter {
    /// Audio accumulated since the first voiced block; `None` until one is seen.
    since_first_voiced_s: Option<f32>,
    trailing_silence_s: f32,
    /// Samples left over from the last push (less than one 100 ms block).
    pending: Vec<f32>,
    #[cfg(test)]
    blocks_analysed: usize,
}

impl SpeechMeter {
    fn new() -> Self {
        SpeechMeter {
            since_first_voiced_s: None,
            trailing_silence_s: 0.0,
            pending: Vec::with_capacity(BLOCK_SAMPLES),
            #[cfg(test)]
            blocks_analysed: 0,
        }
    }

    /// Seconds of speech-bearing audio: the span between the first and the last
    /// voiced block. Exactly `0.0` when no block ever crossed `SILENCE_RMS`.
    fn voiced_s(&self) -> f32 {
        match self.since_first_voiced_s {
            None => 0.0,
            Some(span) => (span - self.trailing_silence_s).max(0.0),
        }
    }

    /// Accounts for `new` samples, consuming them one full 100 ms block at a
    /// time. A partial block is kept for the next call, so the result does not
    /// depend on how the audio was split across polls.
    fn push(&mut self, new: &[f32]) {
        self.pending.extend_from_slice(new);
        let mut consumed = 0;
        while consumed + BLOCK_SAMPLES <= self.pending.len() {
            let block = &self.pending[consumed..consumed + BLOCK_SAMPLES];
            let rms = (block.iter().map(|s| s * s).sum::<f32>() / BLOCK_SAMPLES as f32).sqrt();
            if rms >= SILENCE_RMS {
                // First voiced block opens the span; the rest extends it.
                *self.since_first_voiced_s.get_or_insert(0.0) += BLOCK_S;
                self.trailing_silence_s = 0.0;
            } else {
                // Silence before any speech is not part of the span at all.
                if let Some(span) = self.since_first_voiced_s.as_mut() {
                    *span += BLOCK_S;
                }
                self.trailing_silence_s += BLOCK_S;
            }
            consumed += BLOCK_SAMPLES;
            #[cfg(test)]
            {
                self.blocks_analysed += 1;
            }
        }
        self.pending.drain(..consumed);
    }

    /// Call whenever the buffer it measures is emptied or truncated.
    fn reset(&mut self) {
        self.since_first_voiced_s = None;
        self.trailing_silence_s = 0.0;
        self.pending.clear();
    }
}

fn run(
    drain: DrainHandle,
    transcriber: Arc<Mutex<Option<Transcriber>>>,
    params: StreamParams,
    emit: &(impl Fn(&str) + Send),
    stop: Arc<AtomicBool>,
    cancel: Arc<AtomicBool>,
    tail_rx: Receiver<Vec<f32>>,
) -> Result<String, String> {
    let mut buf: Vec<f32> = Vec::new();
    let mut text = String::new();
    // Normalised form of the last emitted chunk, to drop a chunk that merely
    // repeats the previous one (Whisper echoes the prompt tail on low-content
    // chunks during hesitations/pauses).
    let mut last_norm = String::new();

    let cancelled = &cancel;
    let flush = |buf: &mut Vec<f32>, text: &mut String, last_norm: &mut String| -> Result<(), String> {
        let mut chunk = std::mem::take(buf);
        // Applied to the whole chunk, never per poll: `boost_quiet` is a peak
        // normaliser, so running it on each 200 ms slice would give every slice
        // its own gain — a staircase of amplification that lifts the noise
        // floor between words. One gain per utterance, exactly as the one-shot
        // path does after `recorder.stop()`.
        if params.low_voice {
            crate::audio::boost_quiet(&mut chunk);
        }
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
        let raw = t.transcribe(&chunk, params.language.as_deref(), false, prompt_opt, params.beam_size, None)?;
        // Collapse a phrase repeated inside the same chunk ("X X X" -> "X").
        let piece = collapse_repeats(&raw);
        if piece.is_empty() {
            return Ok(());
        }
        // Drop a chunk that is an exact (normalised) repeat of the previous one.
        let norm = normalize(&piece);
        if norm == *last_norm {
            return Ok(());
        }
        *last_norm = norm;
        // Cancelled while this chunk was being transcribed: drop it instead
        // of pasting into an app the user already backed out of.
        if cancelled.load(Ordering::Relaxed) {
            return Ok(());
        }
        let out = if text.is_empty() { piece } else { format!(" {piece}") };
        text.push_str(&out);
        emit(&out);
        Ok(())
    };

    let mut meter = SpeechMeter::new();

    while !stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(POLL_MS));
        let new = drain.drain();
        meter.push(&new);
        buf.extend(new);
        let buffered_s = buf.len() as f32 / SAMPLE_RATE as f32;
        let voiced_s = meter.voiced_s();

        // Forced cut: the buffer cannot be allowed to grow without end.
        if buffered_s >= MAX_CHUNK_S {
            if voiced_s >= MIN_TAIL_VOICED_S {
                // There is speech in there. Transcribe it — never discard it,
                // whatever `MIN_VOICED_S` would have said about a pause cut.
                if let Err(e) = flush(&mut buf, &mut text, &mut last_norm) {
                    eprintln!("[streaming] chunk: {e}");
                }
                meter.reset();
            } else {
                // Fifteen seconds that never crossed the threshold: room noise.
                // Drop it rather than carry it to the end of the take, keeping
                // the last second in case a word has only just started.
                let keep = SAMPLE_RATE.min(buf.len());
                buf.drain(..buf.len() - keep);
                meter.reset();
                meter.push(&buf);
            }
            continue;
        }

        // Normal cut, on a pause long enough to end a sentence.
        if voiced_s >= MIN_VOICED_S && meter.trailing_silence_s >= PAUSE_S {
            if let Err(e) = flush(&mut buf, &mut text, &mut last_norm) {
                eprintln!("[streaming] chunk: {e}");
            }
            // `flush` empties the buffer whatever it decides to do with the text.
            meter.reset();
        }
    }

    // Remainder: the final audio returned by recorder.stop() on the caller side.
    if let Ok(tail) = tail_rx.recv_timeout(Duration::from_secs(5)) {
        meter.push(&tail);
        buf.extend(tail);
    }
    // Cancelled take: skip the remainder entirely (no transcription, no paste)
    // and report nothing, so the caller does not log it in the history.
    if cancel.load(Ordering::Relaxed) {
        return Ok(String::new());
    }
    if meter.voiced_s() >= MIN_TAIL_VOICED_S {
        flush(&mut buf, &mut text, &mut last_norm)?;
    }
    Ok(text)
}

/// Lowercased, punctuation-stripped, whitespace-collapsed form for comparing
/// two transcribed pieces (used to drop a chunk that repeats the previous one).
fn normalize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_alphanumeric() || c.is_whitespace() { c } else { ' ' })
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Collapses an immediately repeated multi-word block inside a single piece:
/// `"a b c a b c a b c"` -> `"a b c"`. Only blocks of >= 2 words are collapsed
/// so genuine single-word stutters ("le chat le chien") are preserved. Words
/// are compared on their lowercased alphanumeric core, so punctuation/casing
/// differences ("dégradé," vs "dégradé") still count as a repeat.
fn collapse_repeats(s: &str) -> String {
    let words: Vec<&str> = s.split_whitespace().collect();
    if words.len() < 4 {
        return s.trim().to_string();
    }
    let key: Vec<String> = words
        .iter()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .flat_map(|c| c.to_lowercase())
                .collect()
        })
        .collect();

    let mut out: Vec<usize> = Vec::with_capacity(words.len());
    for i in 0..words.len() {
        out.push(i);
        loop {
            let n = out.len();
            let mut collapsed = false;
            // Largest block first so "x x x x" collapses fully.
            for b in (2..=n / 2).rev() {
                if (0..b).all(|k| key[out[n - b + k]] == key[out[n - 2 * b + k]]) {
                    out.truncate(n - b);
                    collapsed = true;
                    break;
                }
            }
            if !collapsed {
                break;
            }
        }
    }
    out.into_iter().map(|i| words[i]).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` seconds of audio at `amp` amplitude (0.0 = silence).
    fn tone(secs: f32, amp: f32) -> Vec<f32> {
        let n = (SAMPLE_RATE as f32 * secs) as usize;
        (0..n).map(|i| if i % 2 == 0 { amp } else { -amp }).collect()
    }

    #[test]
    fn meter_is_independent_of_how_audio_is_split() {
        // The whole point of the incremental meter: feeding the same audio in
        // one go or poll by poll must give the same measurement.
        let audio = [tone(1.0, 0.5), tone(1.5, 0.0)].concat();

        let mut whole = SpeechMeter::new();
        whole.push(&audio);

        let mut chunked = SpeechMeter::new();
        for part in audio.chunks(SAMPLE_RATE / 5) {
            chunked.push(part);
        }

        assert!((whole.voiced_s() - chunked.voiced_s()).abs() < 1e-4);
        assert!((whole.trailing_silence_s - chunked.trailing_silence_s).abs() < 1e-4);
        assert!((whole.voiced_s() - 1.0).abs() < 0.15, "voiced={}", whole.voiced_s());
        assert!(
            (whole.trailing_silence_s - 1.5).abs() < 0.15,
            "silence={}",
            whole.trailing_silence_s
        );
    }

    #[test]
    fn meter_reports_no_voiced_audio_on_pure_silence() {
        // Anti-regression: simply capping the trailing-silence scan would make
        // `voiced` = duration - capped_silence, so a long silent buffer would
        // clear MIN_VOICED_S and be sent to transcription.
        let mut m = SpeechMeter::new();
        for _ in 0..300 {
            m.push(&tone(1.0, 0.0)); // 5 minutes of silence
        }
        assert_eq!(m.voiced_s(), 0.0);
        assert!(m.voiced_s() < MIN_VOICED_S);
        assert!(m.trailing_silence_s >= PAUSE_S);
    }

    #[test]
    fn meter_cost_stays_linear_on_a_long_silence() {
        // Each 100 ms block must be analysed exactly once, however long the
        // take runs. Rescanning the buffer every poll was quadratic (BUG-002).
        let mut m = SpeechMeter::new();
        let poll = tone(0.2, 0.0); // one 200 ms poll
        for _ in 0..1500 {
            m.push(&poll); // 5 minutes
        }
        assert_eq!(m.blocks_analysed, 3000, "un bloc doit etre analyse une seule fois");
    }

    #[test]
    fn meter_silence_resets_after_speech_resumes() {
        let mut m = SpeechMeter::new();
        m.push(&tone(0.5, 0.5));
        m.push(&tone(1.0, 0.0));
        assert!(m.trailing_silence_s >= 0.9);
        m.push(&tone(0.3, 0.5));
        assert_eq!(m.trailing_silence_s, 0.0, "la parole doit remettre le silence a zero");
    }

    /// Realistic speech: words above the threshold separated by the
    /// sub-threshold gaps every speaker leaves between them. 200 ms of word for
    /// 300 ms of gap — deliberately gap-heavy, which is where the regression lived.
    fn speech(words: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..words {
            out.extend(tone(0.2, 0.05)); // a word
            out.extend(tone(0.3, 0.001)); // the gap after it
        }
        out
    }

    /// Total duration of the blocks that are individually above the threshold —
    /// i.e. what the regressed implementation used as `voiced_s`.
    fn loud_block_total(samples: &[f32]) -> f32 {
        samples
            .chunks_exact(BLOCK_SAMPLES)
            .filter(|b| {
                (b.iter().map(|s| s * s).sum::<f32>() / BLOCK_SAMPLES as f32).sqrt() >= SILENCE_RMS
            })
            .count() as f32
            * BLOCK_S
    }

    #[test]
    fn a_short_utterance_reaches_the_flush_threshold() {
        // The regression: summing only the loud blocks made `voiced_s` grow far
        // slower than MIN_VOICED_S was calibrated for, so a short sentence never
        // armed a flush, nothing was ever transcribed, and the take came back
        // empty. Two words span 0.7 s and must clear the 0.6 s gate.
        let audio = speech(2);
        let mut m = SpeechMeter::new();
        m.push(&audio);

        // This is what makes the test a regression test rather than a
        // tautology: the discarded implementation is measured here too, and it
        // provably fails the gate on the very same audio.
        assert!(
            loud_block_total(&audio) < MIN_VOICED_S,
            "l'audio doit piéger l'ancienne mesure, sinon ce test ne prouve rien"
        );
        assert!(
            m.voiced_s() >= MIN_VOICED_S,
            "deux mots doivent armer le flush, voiced={}",
            m.voiced_s()
        );
    }

    #[test]
    fn a_pause_after_speech_still_cuts_the_chunk() {
        // The span must not keep growing through the trailing silence, or the
        // pause would never be seen as a pause.
        let mut m = SpeechMeter::new();
        m.push(&speech(3));
        let before = m.voiced_s();
        m.push(&tone(PAUSE_S + 0.3, 0.0));
        assert!(m.trailing_silence_s >= PAUSE_S, "la pause doit etre vue");
        assert!(
            (m.voiced_s() - before).abs() < 0.05,
            "le silence final ne doit pas gonfler la parole: {before} -> {}",
            m.voiced_s()
        );
    }

    #[test]
    fn leading_silence_is_not_counted_as_speech() {
        // Better than the formula this replaces, which counted everything
        // before the last voiced block - including silence before the first.
        let mut m = SpeechMeter::new();
        m.push(&tone(5.0, 0.0));
        assert_eq!(m.voiced_s(), 0.0);
        m.push(&tone(0.4, 0.05));
        assert!(
            (m.voiced_s() - 0.4).abs() < 0.15,
            "seule la parole compte, voiced={}",
            m.voiced_s()
        );
    }

    #[test]
    fn collapse_repeated_phrase() {
        let s = "DMOT score partiellement dégradé, DMOT score partiellement dégradé, \
                 DMOT score partiellement dégradé";
        assert_eq!(collapse_repeats(s), "DMOT score partiellement dégradé,");
    }

    #[test]
    fn collapse_preserves_non_repeats() {
        assert_eq!(collapse_repeats("le chat et le chien"), "le chat et le chien");
    }

    #[test]
    fn collapse_keeps_short_pieces() {
        assert_eq!(collapse_repeats("oui oui"), "oui oui");
    }

    #[test]
    fn normalize_matches_punctuation_variants() {
        assert_eq!(normalize("Dégradé,"), normalize("dégradé"));
    }
}
