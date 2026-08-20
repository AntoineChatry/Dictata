//! Optional audio conversion helper (feature `resample`).
//!
//! Every backend consumes the same contract — **mono, 16 kHz, `f32`** in
//! `[-1.0, 1.0]` (see [`crate::engine::SAMPLE_RATE`]). Capture hardware and decoded media
//! rarely match it: microphones deliver 44.1/48 kHz, files come stereo. This
//! module bridges the gap with **channel averaging + linear interpolation** —
//! pure Rust, no extra dependency, negligible CPU (a few flops per output
//! sample). Linear resampling is coarser than a windowed-sinc resampler, but for
//! 16 kHz ASR the difference in word-error rate is not measurable, so the
//! dependency-free path wins.
//!
//! It is a *helper*, not part of the [`crate::engine::AsrEngine`] contract: the engine
//! still refuses to guess and never resamples on its own. Call this at the
//! capture/decode stage, then hand the result to `transcribe`.

/// Convert interleaved `channels`-channel PCM sampled at `from_rate` Hz into the
/// engine's **mono 16 kHz** `f32` buffer.
///
/// `input` is interleaved by frame (`[l, r, l, r, …]` for stereo). Channels are
/// averaged to mono, then linearly resampled to [`crate::engine::SAMPLE_RATE`]. When
/// `from_rate` already equals the target and the input is mono, the samples are
/// returned unchanged (a plain copy).
///
/// Returns an empty vector for degenerate input (`channels == 0`, `from_rate == 0`,
/// or empty `input`) rather than panicking — a caller feeding a silent/absent
/// buffer gets an empty result, consistent with `transcribe`'s empty-audio case.
///
/// The output is **not** range-sanitized here; the engines clamp on entry. This
/// keeps the helper a pure format conversion.
pub fn to_engine_audio(input: &[f32], from_rate: u32, channels: u16) -> Vec<f32> {
    if input.is_empty() || channels == 0 || from_rate == 0 {
        return Vec::new();
    }
    let mono = downmix_to_mono(input, channels);
    resample_linear(&mono, from_rate, crate::engine::types::SAMPLE_RATE)
}

/// Average interleaved channels into a mono buffer. A mono input is returned as a
/// direct copy. A trailing partial frame (input length not a multiple of
/// `channels`) is dropped — it cannot form a complete frame.
fn downmix_to_mono(input: &[f32], channels: u16) -> Vec<f32> {
    let channels = channels as usize;
    if channels == 1 {
        return input.to_vec();
    }
    let inv = 1.0 / channels as f32;
    input
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * inv)
        .collect()
}

/// Linearly resample a mono buffer from `from_rate` to `to_rate`.
///
/// Returns a direct copy when the rates match. Otherwise each output sample is
/// interpolated between its two neighbouring input samples; the final sample
/// holds the last input value (no out-of-bounds read).
fn resample_linear(mono: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return mono.to_vec();
    }
    // Output length scaled by the rate ratio, computed in u64 to avoid overflow
    // on long clips (an hour at 48 kHz is ~1.7e8 samples).
    let out_len = (mono.len() as u64 * to_rate as u64 / from_rate as u64) as usize;
    let ratio = from_rate as f64 / to_rate as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src = i as f64 * ratio;
        let idx = src as usize;
        let frac = (src - idx as f64) as f32;
        let a = mono[idx];
        // Clamp the right neighbour to the last sample at the tail.
        let b = mono.get(idx + 1).copied().unwrap_or(a);
        out.push(a + (b - a) * frac);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::types::SAMPLE_RATE;

    #[test]
    fn degenerate_input_yields_empty() {
        assert!(to_engine_audio(&[], 48_000, 2).is_empty());
        assert!(to_engine_audio(&[0.1, 0.2], 48_000, 0).is_empty());
        assert!(to_engine_audio(&[0.1, 0.2], 0, 1).is_empty());
    }

    #[test]
    fn mono_at_target_rate_is_unchanged() {
        let input = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let out = to_engine_audio(&input, SAMPLE_RATE, 1);
        assert_eq!(out, input);
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        // Frames: (0.0,1.0)->0.5, (0.2,0.4)->0.3, (-1.0,1.0)->0.0
        let input = vec![0.0, 1.0, 0.2, 0.4, -1.0, 1.0];
        let out = to_engine_audio(&input, SAMPLE_RATE, 2);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.3).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn trailing_partial_stereo_frame_is_dropped() {
        // 5 samples, 2 channels -> 2 complete frames, last lone sample dropped.
        let input = vec![0.0, 0.0, 0.4, 0.6, 0.9];
        let out = to_engine_audio(&input, SAMPLE_RATE, 2);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn downsampling_scales_length_by_the_rate_ratio() {
        // 32 kHz -> 16 kHz halves the sample count.
        let input: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = to_engine_audio(&input, 32_000, 1);
        assert_eq!(out.len(), 1600);
    }

    #[test]
    fn upsampling_scales_length_by_the_rate_ratio() {
        // 8 kHz -> 16 kHz doubles the sample count.
        let input: Vec<f32> = (0..800).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = to_engine_audio(&input, 8_000, 1);
        assert_eq!(out.len(), 1600);
    }

    #[test]
    fn interpolated_values_stay_within_the_input_envelope() {
        // Linear interpolation never overshoots the neighbouring samples, so the
        // output range cannot exceed the input range.
        let input = vec![0.0, 0.8, -0.4, 0.2, -0.9, 0.6];
        let out = to_engine_audio(&input, 44_100, 1);
        let (lo, hi) = (-0.9f32, 0.8f32);
        assert!(out.iter().all(|&s| s >= lo - 1e-6 && s <= hi + 1e-6));
    }
}
