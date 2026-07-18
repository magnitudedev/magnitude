use base64::Engine;
use base64::engine::general_purpose::STANDARD;

pub(crate) const MAX_HTTP_IMAGE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DecodedImage {
    pub(crate) media_type: String,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum ImageDataUrlError {
    #[error("image_url.url must be a base64 data URL; network URLs are not supported")]
    NotDataUrl,
    #[error("image data URL must declare an image media type")]
    InvalidMediaType,
    #[error("image data URL must use base64 encoding")]
    UnsupportedEncoding,
    #[error("image data URL exceeds the {maximum} byte decoded-size limit")]
    TooLarge { maximum: usize },
    #[error("image data URL contains invalid base64")]
    InvalidBase64,
    #[error("image data URL decoded to an empty payload")]
    Empty,
}

pub(crate) fn decode_image_data_url(
    value: &str,
    maximum_bytes: usize,
) -> Result<DecodedImage, ImageDataUrlError> {
    let data = value
        .strip_prefix("data:")
        .ok_or(ImageDataUrlError::NotDataUrl)?;
    let (metadata, encoded) = data
        .split_once(',')
        .ok_or(ImageDataUrlError::UnsupportedEncoding)?;
    let mut metadata = metadata.split(';');
    let media_type = metadata.next().unwrap_or_default();
    if !media_type.starts_with("image/") || media_type.len() == "image/".len() {
        return Err(ImageDataUrlError::InvalidMediaType);
    }
    if metadata.next() != Some("base64") || metadata.next().is_some() {
        return Err(ImageDataUrlError::UnsupportedEncoding);
    }

    // Reject oversized inputs before allocating their decoded representation. The extra two bytes
    // account for base64 padding in the conservative decoded-length estimate.
    let estimated = encoded
        .len()
        .checked_add(3)
        .and_then(|length| length.checked_div(4))
        .and_then(|groups| groups.checked_mul(3))
        .ok_or(ImageDataUrlError::TooLarge {
            maximum: maximum_bytes,
        })?;
    if estimated > maximum_bytes.saturating_add(2) {
        return Err(ImageDataUrlError::TooLarge {
            maximum: maximum_bytes,
        });
    }

    let bytes = STANDARD
        .decode(encoded)
        .map_err(|_| ImageDataUrlError::InvalidBase64)?;
    if bytes.is_empty() {
        return Err(ImageDataUrlError::Empty);
    }
    if bytes.len() > maximum_bytes {
        return Err(ImageDataUrlError::TooLarge {
            maximum: maximum_bytes,
        });
    }
    Ok(DecodedImage {
        media_type: media_type.to_owned(),
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_strict_base64_image_data_urls() {
        let image = decode_image_data_url("data:image/png;base64,iVBORw==", 32).unwrap();
        assert_eq!(image.media_type, "image/png");
        assert_eq!(image.bytes, b"\x89PNG");
    }

    #[test]
    fn rejects_network_and_non_image_inputs() {
        assert_eq!(
            decode_image_data_url("https://example.invalid/image.png", 32),
            Err(ImageDataUrlError::NotDataUrl)
        );
        assert_eq!(
            decode_image_data_url("data:text/plain;base64,aGk=", 32),
            Err(ImageDataUrlError::InvalidMediaType)
        );
    }

    #[test]
    fn checks_the_decoded_bound_before_and_after_decode() {
        assert_eq!(
            decode_image_data_url("data:image/png;base64,AQIDBA==", 3),
            Err(ImageDataUrlError::TooLarge { maximum: 3 })
        );
    }
}
