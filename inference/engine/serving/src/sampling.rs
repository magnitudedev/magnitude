//! The sampling every protocol serves: one set of defaults and one
//! validation, whichever API a request arrives through.

use magnitude_chat::request::SamplingControls;

use crate::error::ApiError;

const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_TOP_P: f32 = 0.95;
const DEFAULT_TOP_K: u32 = 20;

/// The sampling fields a request may set; an unset field takes the default.
#[derive(Default)]
pub(crate) struct SamplingRequest {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<u32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repetition_penalty: Option<f32>,
    pub(crate) presence_penalty: Option<f32>,
    pub(crate) frequency_penalty: Option<f32>,
    pub(crate) seed: Option<u32>,
}

fn finite_in(value: f32, range: std::ops::RangeInclusive<f32>) -> bool {
    value.is_finite() && range.contains(&value)
}

/// Validated sampling controls with every unset field at its default.
pub(crate) fn sampling(request: SamplingRequest) -> Result<SamplingControls, ApiError> {
    let temperature = request.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    if !finite_in(temperature, 0.0..=2.0) {
        return Err(ApiError::invalid(
            "temperature must be finite and between 0 and 2",
        ));
    }
    let top_p = request.top_p.unwrap_or(DEFAULT_TOP_P);
    if !finite_in(top_p, 0.0..=1.0) || top_p == 0.0 {
        return Err(ApiError::invalid(
            "top_p must be finite, greater than 0 and at most 1",
        ));
    }
    let top_k = request.top_k.unwrap_or(DEFAULT_TOP_K);
    if top_k > 1 << 24 {
        return Err(ApiError::invalid("top_k exceeds the supported range"));
    }
    let min_p = request.min_p.unwrap_or(0.0);
    if !finite_in(min_p, 0.0..=1.0) {
        return Err(ApiError::invalid("min_p must be finite and between 0 and 1"));
    }
    let repetition_penalty = request.repetition_penalty.unwrap_or(1.0);
    if !repetition_penalty.is_finite() || repetition_penalty <= 0.0 {
        return Err(ApiError::invalid(
            "repetition_penalty must be finite and positive",
        ));
    }
    let presence_penalty = request.presence_penalty.unwrap_or(0.0);
    let frequency_penalty = request.frequency_penalty.unwrap_or(0.0);
    if !finite_in(presence_penalty, -2.0..=2.0) || !finite_in(frequency_penalty, -2.0..=2.0) {
        return Err(ApiError::invalid(
            "presence_penalty and frequency_penalty must be finite and between -2 and 2",
        ));
    }
    Ok(SamplingControls {
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        seed: request_seed(request.seed),
    })
}

/// The seed a request samples with: the caller's, or a fresh one, so an
/// identical request without a seed samples anew rather than repeating
/// the same draws (and the same failure) on every retry.
fn request_seed(seed: Option<u32>) -> u64 {
    seed.map(u64::from).unwrap_or_else(|| {
        use std::hash::{BuildHasher, Hasher};
        std::collections::hash_map::RandomState::new().build_hasher().finish()
    })
}
