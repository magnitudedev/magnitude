import { Image, format } from '@magnitudedev/image'
import type { ImageMediaType } from '@magnitudedev/tools'

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
  options?: ReadImageFileOptions
): Promise<ReadImageFileResult> {
  const maxLongEdge = options?.maxLongEdge ?? DEFAULT_MAX_LONG_EDGE
  const jpegQuality = options?.jpegQuality ?? 80

  const fileBuffer = await Bun.file(absolutePath).arrayBuffer()
  const buf = new Uint8Array(fileBuffer)

  const detectedFormat = format(buf)

  if (detectedFormat === 'svg' || absolutePath.endsWith('.svg')) {
    const img = Image.fromSvg(buf, { maxWidth: maxLongEdge, maxHeight: maxLongEdge })
    const pngBuffer = img.toPng()
    return {
      base64: pngBuffer.toString('base64'),
      mediaType: 'image/png',
      width: img.width,
      height: img.height,
    }
  }

  let img = Image.fromBuffer(buf)

  const longEdge = Math.max(img.width, img.height)
  if (longEdge > maxLongEdge) {
    const scale = maxLongEdge / longEdge
    img = img.resize(
      Math.max(1, Math.round(img.width * scale)),
      Math.max(1, Math.round(img.height * scale))
    )
  }

  const mediaType = IMAGE_MEDIA_TYPES[detectedFormat] ?? 'image/png'

  let encoded: Buffer
  let outputMediaType: ImageMediaType

  if (mediaType === 'image/jpeg') {
    encoded = img.toJpeg(jpegQuality)
    outputMediaType = 'image/jpeg'
  } else {
    encoded = img.toPng()
    outputMediaType = 'image/png'
  }

  if (encoded.length > MAX_ENCODED_BYTES && outputMediaType !== 'image/jpeg') {
    encoded = img.toJpeg(jpegQuality)
    outputMediaType = 'image/jpeg'
  }

  if (encoded.length > MAX_ENCODED_BYTES) {
    for (const reducedEdge of [1280, 1024, 768]) {
      const scale = reducedEdge / Math.max(img.width, img.height)
      if (scale >= 1) continue

      const reduced = img.resize(
        Math.max(1, Math.round(img.width * scale)),
        Math.max(1, Math.round(img.height * scale))
      )
      encoded = reduced.toJpeg(70)
      outputMediaType = 'image/jpeg'
      img = reduced

      if (encoded.length <= MAX_ENCODED_BYTES) break
    }
  }

  return {
    base64: encoded.toString('base64'),
    mediaType: outputMediaType,
    width: img.width,
    height: img.height,
  }
}