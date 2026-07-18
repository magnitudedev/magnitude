use std::ffi::CString;

use getrandom::fill;
use icn_core::{
    FlashAttention, ImageInput, ImageInputLimits, InferenceError, ModelModalities, ProjectorConfig,
};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::FlashAttentionPolicy;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputChunkType, MtmdInputChunks, MtmdInputText,
};
use llama_cpp_2::token::LlamaToken;
use sha2::{Digest, Sha256};

pub(crate) struct MultimodalRuntime<'model> {
    context: MtmdContext<'model>,
    marker: String,
    limits: ImageInputLimits,
    modalities: ModelModalities,
}

pub(crate) struct MultimodalPrompt {
    chunks: MtmdInputChunks,
    text_tokens: Vec<LlamaToken>,
    total_tokens: usize,
}

impl MultimodalPrompt {
    pub(crate) fn text_tokens(&self) -> &[LlamaToken] {
        &self.text_tokens
    }

    pub(crate) const fn total_tokens(&self) -> usize {
        self.total_tokens
    }
}

impl<'model> MultimodalRuntime<'model> {
    pub(crate) fn load(
        config: &ProjectorConfig,
        model: &'model LlamaModel,
        flash_attention: FlashAttention,
        threads: Option<i32>,
    ) -> Result<Self, InferenceError> {
        let marker = random_media_marker()?;
        let path = config.path.to_str().ok_or_else(|| {
            InferenceError::InvalidConfig(format!(
                "multimodal projector path is not valid UTF-8: {}",
                config.path.display()
            ))
        })?;
        let params = MtmdContextParams {
            use_gpu: config.use_gpu,
            print_timings: false,
            n_threads: threads.unwrap_or(4),
            media_marker: CString::new(marker.as_str()).map_err(native_error)?,
            flash_attention: match flash_attention {
                FlashAttention::Auto => FlashAttentionPolicy::Auto,
                FlashAttention::Disabled => FlashAttentionPolicy::Disabled,
                FlashAttention::Enabled => FlashAttentionPolicy::Enabled,
            },
            warmup: config.warmup,
            image_min_tokens: config.image_min_tokens,
            image_max_tokens: config.image_max_tokens,
        };
        let context = MtmdContext::init_from_file(path, model, &params).map_err(native_error)?;
        let modalities = ModelModalities {
            vision: context.support_vision(),
            // ICN's current wire contract accepts images only. Do not claim native projector
            // capabilities that requests cannot actually reach yet.
            audio: false,
            video: false,
        };
        if !modalities.vision {
            return Err(InferenceError::InvalidConfig(
                "the configured multimodal projector does not support image input".into(),
            ));
        }
        Ok(Self {
            context,
            marker,
            limits: config.input_limits,
            modalities,
        })
    }

    pub(crate) fn marker(&self) -> &str {
        &self.marker
    }

    pub(crate) const fn modalities(&self) -> ModelModalities {
        self.modalities
    }

    pub(crate) fn prepare_prompt(
        &self,
        prompt: String,
        images: &[ImageInput],
    ) -> Result<MultimodalPrompt, InferenceError> {
        if images.is_empty() {
            return Err(InferenceError::InvalidConfig(
                "multimodal prompt preparation requires at least one image".into(),
            ));
        }
        if images.len() > self.limits.max_images.get() as usize {
            return Err(InferenceError::InvalidConfig(format!(
                "request contains {} images; the configured limit is {}",
                images.len(),
                self.limits.max_images
            )));
        }
        let marker_count = prompt.match_indices(&self.marker).count();
        if marker_count != images.len() {
            return Err(InferenceError::Backend(format!(
                "prepared prompt contains {marker_count} media markers for {} images",
                images.len()
            )));
        }

        // Placeholder decoding reads media dimensions without retaining decoded pixels. This lets
        // us reject decompression bombs before native preprocessing allocates the full RGB image.
        let mut decoded_total = 0usize;
        for (index, image) in images.iter().enumerate() {
            validate_image_envelope(image, index, self.limits)?;
            let placeholder = MtmdBitmap::from_buffer(&self.context, image.bytes(), true)
                .map_err(|error| invalid_image(index, error))?;
            if placeholder.is_audio() {
                return Err(InferenceError::InvalidConfig(format!(
                    "image {index} decoded as audio; only image input is supported"
                )));
            }
            let decoded_bytes = decoded_rgb_bytes(&placeholder, index)?;
            if decoded_bytes > self.limits.max_decoded_bytes_per_image.get() {
                return Err(InferenceError::InvalidConfig(format!(
                    "image {index} requires {decoded_bytes} decoded RGB bytes; the per-image limit is {}",
                    self.limits.max_decoded_bytes_per_image
                )));
            }
            decoded_total = decoded_total.checked_add(decoded_bytes).ok_or_else(|| {
                InferenceError::InvalidConfig("aggregate decoded image size overflowed".into())
            })?;
            if decoded_total > self.limits.max_total_decoded_bytes.get() {
                return Err(InferenceError::InvalidConfig(format!(
                    "images require {decoded_total} decoded RGB bytes; the request limit is {}",
                    self.limits.max_total_decoded_bytes
                )));
            }
        }

        let mut bitmaps = Vec::with_capacity(images.len());
        for (index, image) in images.iter().enumerate() {
            let mut bitmap = MtmdBitmap::from_buffer(&self.context, image.bytes(), false)
                .map_err(|error| invalid_image(index, error))?;
            if bitmap.is_audio() {
                return Err(InferenceError::InvalidConfig(format!(
                    "image {index} decoded as audio; only image input is supported"
                )));
            }
            let expected = decoded_rgb_bytes(&bitmap, index)?;
            let data = bitmap.data().map_err(|error| invalid_image(index, error))?;
            if data.len() != expected {
                return Err(InferenceError::Backend(format!(
                    "image {index} decoded to {} bytes but its dimensions require {expected}",
                    data.len()
                )));
            }
            bitmap
                .set_id(&format!("{:x}", Sha256::digest(data)))
                .map_err(native_error)?;
            bitmaps.push(bitmap);
        }

        let bitmap_refs = bitmaps.iter().collect::<Vec<_>>();
        let chunks = self
            .context
            .tokenize(
                MtmdInputText {
                    text: prompt,
                    add_special: true,
                    parse_special: true,
                },
                &bitmap_refs,
            )
            .map_err(native_error)?;
        let total_tokens = chunks.total_tokens().map_err(native_error)?;
        if chunks.is_empty() || total_tokens == 0 {
            return Err(InferenceError::InvalidConfig(
                "the prepared multimodal prompt produced no tokens".into(),
            ));
        }
        if chunks.last_chunk_type() != Some(MtmdInputChunkType::Text) {
            return Err(InferenceError::InvalidConfig(
                "a multimodal chat prompt must end in text so generation logits are available"
                    .into(),
            ));
        }
        let text_tokens = chunks.text_tokens().map_err(native_error)?;
        Ok(MultimodalPrompt {
            chunks,
            text_tokens,
            total_tokens,
        })
    }

    pub(crate) fn evaluate_prompt(
        &mut self,
        prompt: &MultimodalPrompt,
        llama_context: &mut LlamaContext<'_>,
        sequence_id: i32,
        batch_size: i32,
    ) -> Result<i32, InferenceError> {
        prompt
            .chunks
            .eval_chunks(
                &mut self.context,
                llama_context,
                0,
                sequence_id,
                batch_size,
                true,
            )
            .map_err(native_error)
    }
}

fn validate_image_envelope(
    image: &ImageInput,
    index: usize,
    limits: ImageInputLimits,
) -> Result<(), InferenceError> {
    if !image.media_type().starts_with("image/") {
        return Err(InferenceError::InvalidConfig(format!(
            "image {index} has non-image media type {}",
            image.media_type()
        )));
    }
    if image.bytes().is_empty() {
        return Err(InferenceError::InvalidConfig(format!(
            "image {index} has no bytes"
        )));
    }
    if image.bytes().len() > limits.max_input_bytes_per_image.get() {
        return Err(InferenceError::InvalidConfig(format!(
            "image {index} contains {} compressed bytes; the per-image limit is {}",
            image.bytes().len(),
            limits.max_input_bytes_per_image
        )));
    }
    Ok(())
}

fn decoded_rgb_bytes(bitmap: &MtmdBitmap, index: usize) -> Result<usize, InferenceError> {
    if bitmap.nx() == 0 || bitmap.ny() == 0 {
        return Err(InferenceError::InvalidConfig(format!(
            "image {index} has zero width or height"
        )));
    }
    (bitmap.nx() as usize)
        .checked_mul(bitmap.ny() as usize)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| InferenceError::InvalidConfig(format!("image {index} dimensions overflow")))
}

fn random_media_marker() -> Result<String, InferenceError> {
    let mut random = [0u8; 16];
    fill(&mut random).map_err(native_error)?;
    Ok(format!("<__magnitude_media_{}__>", hex(&random)))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn invalid_image(index: usize, error: impl std::fmt::Display) -> InferenceError {
    InferenceError::InvalidConfig(format!("image {index} could not be decoded: {error}"))
}

fn native_error(error: impl std::fmt::Display) -> InferenceError {
    InferenceError::Backend(error.to_string())
}

#[cfg(test)]
mod tests {
    use std::num::{NonZeroU32, NonZeroUsize};

    use super::*;

    fn limits() -> ImageInputLimits {
        ImageInputLimits {
            max_images: NonZeroU32::new(1).unwrap(),
            max_input_bytes_per_image: NonZeroUsize::new(2).unwrap(),
            max_decoded_bytes_per_image: NonZeroUsize::new(3).unwrap(),
            max_total_decoded_bytes: NonZeroUsize::new(3).unwrap(),
        }
    }

    #[test]
    fn envelope_rejects_non_images_empty_images_and_large_inputs() {
        assert!(
            validate_image_envelope(&ImageInput::new("text/plain", vec![1]), 0, limits()).is_err()
        );
        assert!(
            validate_image_envelope(&ImageInput::new("image/png", Vec::<u8>::new()), 0, limits())
                .is_err()
        );
        assert!(
            validate_image_envelope(&ImageInput::new("image/png", vec![1, 2, 3]), 0, limits())
                .is_err()
        );
        validate_image_envelope(&ImageInput::new("image/png", vec![1, 2]), 0, limits()).unwrap();
    }

    #[test]
    fn media_markers_are_process_local_and_unambiguous() {
        let marker = random_media_marker().unwrap();
        assert!(marker.starts_with("<__magnitude_media_"));
        assert!(marker.ends_with("__>"));
        assert_eq!(marker.len(), "<__magnitude_media___>".len() + 32);
    }
}
