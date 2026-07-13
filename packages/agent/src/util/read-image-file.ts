import type { ImageMediaType } from '@magnitudedev/ai'

export interface ReadImageFileOptions {
  readonly maxLongEdge?: number
  readonly jpegQuality?: number
}

export interface ReadImageFileResult {
  readonly base64: string
  readonly mediaType: ImageMediaType
  readonly width: number
  readonly height: number
}

const IMAGE_MEDIA_TYPES: Record<string, ImageMediaType> = {
  png: 'image/png',
  jpeg: 'image/jpeg',
  jpg: 'image/jpeg',
  webp: 'image/webp',
  gif: 'image/gif',
}

const DEFAULT_MAX_LONG_EDGE = 1568
const MAX_ENCODED_BYTES = 8 * 1024 * 1024

export async function readImageFileForModel(
  absolutePath: string,
  options?: ReadImageFileOptions,
): Promise<ReadImageFileResult> {
  const maxLongEdge = options?.maxLongEdge ?? DEFAULT_MAX_LONG_EDGE
  const jpegQuality = options?.jpegQuality ?? 80

  const fileBuffer = await Bun.file(absolutePath).arrayBuffer()
  const buf = new Uint8Array(fileBuffer)

  // Detect format via Bun.Image metadata (no pixel decode needed)
  const meta = await new Bun.Image(buf).metadata().catch(() => null)
  const detectedFormat = meta?.format ?? null

  if (detectedFormat === null && absolutePath.endsWith('.svg')) {
    // Bun.Image does not handle SVG; fallback to original data as base64.
    // SVG files are typically small text files, so we skip resizing.
    return {
      base64: Buffer.from(buf).toString('base64'),
      mediaType: 'image/png',
      width: 0,
      height: 0,
    }
  }

  if (detectedFormat === null) {
    throw new Error(`Unsupported or unknown image format: ${absolutePath}`)
  }

  // Build the pipeline: decode → resize if needed → encode
  let pipeline = new Bun.Image(buf)

  const longEdge = Math.max(meta!.width, meta!.height)
  if (longEdge > maxLongEdge) {
    pipeline = pipeline.resize(maxLongEdge, maxLongEdge, {
      fit: 'inside',
      withoutEnlargement: true,
    })
  }

  const mediaType = IMAGE_MEDIA_TYPES[detectedFormat] ?? 'image/png'

  let encoded: Buffer
  let outputMediaType: ImageMediaType

  if (mediaType === 'image/jpeg') {
    encoded = await pipeline.jpeg({ quality: jpegQuality }).buffer()
    outputMediaType = 'image/jpeg'
  } else {
    encoded = await pipeline.png().buffer()
    outputMediaType = 'image/png'
  }

  if (encoded.length > MAX_ENCODED_BYTES && outputMediaType !== 'image/jpeg') {
    encoded = await pipeline.jpeg({ quality: jpegQuality }).buffer()
    outputMediaType = 'image/jpeg'
  }

  if (encoded.length > MAX_ENCODED_BYTES) {
    // Try progressively smaller sizes
    for (const reducedEdge of [1280, 1024, 768]) {
      const currentLongEdge = Math.max(meta!.width, meta!.height)
      const scale = reducedEdge / currentLongEdge
      if (scale >= 1) continue

      const reduced = new Bun.Image(buf).resize(
        Math.max(1, Math.round(meta!.width * scale)),
        Math.max(1, Math.round(meta!.height * scale)),
        { fit: 'inside', withoutEnlargement: true },
      )
      encoded = await reduced.jpeg({ quality: 70 }).buffer()
      outputMediaType = 'image/jpeg'

      if (encoded.length <= MAX_ENCODED_BYTES) break
    }
  }

  return {
    base64: encoded.toString('base64'),
    mediaType: outputMediaType,
    width: pipeline.width,
    height: pipeline.height,
  }
}