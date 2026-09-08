"""Bounded image-source decoding; neural preprocessing belongs to the model adapter."""

import base64
import io

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGES = 16
MAX_SOURCE_BYTES = 16 << 20
MAX_PIXELS = 16_000_000


def replace_image_parts(messages: list[dict]) -> list[Image.Image]:
    """Replace ordered API image parts with template markers, returning RGB sources.

    This local API accepts inline base64 images. Network fetch and local filesystem
    access are deliberately outside its source-resolution policy.
    """
    images = []
    total = 0
    pixels = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part["type"] == "text":
                continue
            source = part.get("image_url")
            if not isinstance(source, dict) or not {"url"} <= source.keys() <= {"url", "detail"}:
                raise ValueError("image_url requires a URL and optional detail")
            if source.get("detail", "auto") != "auto":
                raise ValueError("this model binding supports image detail=auto")
            url = source["url"]
            if not isinstance(url, str) or not url.startswith("data:image/"):
                raise ValueError("image sources must be inline base64 image data URLs")
            header, separator, encoded = url.partition(",")
            if not separator or not header.endswith(";base64"):
                raise ValueError("image data URL must contain base64 bytes")
            if len(images) >= MAX_IMAGES or len(encoded) > 4 * (
                (MAX_SOURCE_BYTES - total + 2) // 3
            ):
                raise ValueError("request exceeds its image source limit")
            data = base64.b64decode(encoded, validate=True)
            total += len(data)
            if total > MAX_SOURCE_BYTES:
                raise ValueError("request exceeds its image byte limit")
            try:
                with Image.open(io.BytesIO(data)) as image:
                    pixels += image.width * image.height
                    if pixels > MAX_PIXELS or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("images must be single-frame and within the pixel limit")
                    images.append(ImageOps.exif_transpose(image).convert("RGB"))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
                raise ValueError("image source could not be decoded") from error
            content[index] = {"type": "image"}
    return images
