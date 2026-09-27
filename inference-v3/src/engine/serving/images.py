"""Bounded inline image resolution, independent of model preparation."""

import base64
import binascii
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 16
MAX_SOURCE_BYTES = 16 << 20
MAX_PIXELS = 16_000_000


def resolve_images(messages: list[dict]) -> tuple[list[dict], tuple[Image.Image, ...]]:
    """Preserve content order, replacing image sources with template markers.

    The caller's messages are never modified. Only inline base64, single-frame
    images at the bound processor's default quality are supported.
    """
    normalized, images = [], []
    total_bytes = total_pixels = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            normalized.append(dict(message))
            continue
        parts = []
        for part in content:
            if part.get("type") == "text":
                parts.append(dict(part))
                continue
            if part.get("type") != "image_url" or set(part) != {"type", "image_url"}:
                raise ValueError("unsupported message content part")
            source = part["image_url"]
            if not isinstance(source, dict) or not {"url"} <= source.keys() <= {"url", "detail"}:
                raise ValueError("image_url requires a URL and optional detail")
            if source.get("detail", "auto") != "auto":
                raise ValueError("image preparation supports detail=auto")
            url = source["url"]
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("image sources must be inline base64 data URLs")
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("image data URL must contain base64 bytes")
            if len(images) >= MAX_IMAGES or len(encoded) > 4 * (
                (MAX_SOURCE_BYTES - total_bytes + 2) // 3
            ):
                raise ValueError("request exceeds image source limit")
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as error:
                raise ValueError("image source has invalid base64") from error
            total_bytes += len(data)
            if total_bytes > MAX_SOURCE_BYTES:
                raise ValueError("request exceeds image byte limit")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    total_pixels += image.width * image.height
                    if total_pixels > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("images must be single-frame and within pixel limits")
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                raise ValueError("image source could not be decoded") from error
            parts.append({"type": "image"})
        normalized.append({**message, "content": parts})
    return normalized, tuple(images)
