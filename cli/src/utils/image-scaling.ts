import { logger } from '@magnitudedev/logger'
import { Image } from '@magnitudedev/image'

export interface ImagePayload {
  base64: string
  mime: string
  width: number
  height: number
  filename?: string
}

export interface AutoScaleImageResult extends ImagePayload {
  wasScaled: boolean
  originalBytes: number
  finalBytes: number
  selectedQuality?: number
}

const MAX_BASE64_BYTES = 5 * 1024 * 1024
const MIN_DIMENSION = 256

export function estimateBase64Bytes(base64: string): number {
  const len = base64.length
  if (len === 0) return 0
  let padding = 0
  if (base64.endsWith('==')) padding = 2
  else if (base64.endsWith('=')) padding = 1
  return Math.floor((len * 3) / 4) - padding
}

function computeTargetDimensions(width: number, height: number, factor: number): { width: number; height: number } {
  return {
    width: Math.min(width, Math.max(MIN_DIMENSION, Math.floor(width * factor))),
    height: Math.min(height, Math.max(MIN_DIMENSION, Math.floor(height * factor))),
  }
}

export async function autoScaleImageAttachmentIfNeeded(input: ImagePayload): Promise<AutoScaleImageResult> {
  const originalBytes = estimateBase64Bytes(input.base64)
  if (originalBytes <= MAX_BASE64_BYTES) {
    return {
      ...input,
      wasScaled: false,
      originalBytes,
      finalBytes: originalBytes,
    }
  }

  try {
    const sourceBuffer = Buffer.from(input.base64, 'base64')
    const sourceImage = Image.fromBuffer(sourceBuffer)

    let bestCandidate: { buffer: Buffer; width: number; height: number; quality: number } | null = null

    const ratio = MAX_BASE64_BYTES / originalBytes
    const firstFactor = Math.min(1, Math.sqrt(ratio) * 0.85)
    const firstTarget = computeTargetDimensions(input.width, input.height, firstFactor)

    const attempts = [
      { width: firstTarget.width, height: firstTarget.height, quality: 80 },
      {
        width: Math.min(input.width, Math.max(MIN_DIMENSION, Math.floor(firstTarget.width * 0.5))),
        height: Math.min(input.height, Math.max(MIN_DIMENSION, Math.floor(firstTarget.height * 0.5))),
        quality: 65,
      },
    ]

    for (const attempt of attempts) {
      const resized = sourceImage.resize(attempt.width, attempt.height)
      const outputBuffer = resized.toJpeg(attempt.quality)

      if (outputBuffer.length === 0) continue

      const candidate = { buffer: outputBuffer, width: resized.width, height: resized.height, quality: attempt.quality }

      if (!bestCandidate || candidate.buffer.length < bestCandidate.buffer.length) {
        bestCandidate = candidate
      }

      if (outputBuffer.length <= MAX_BASE64_BYTES) {
        return {
          base64: outputBuffer.toString('base64'),
          mime: 'image/jpeg',
          width: resized.width,
          height: resized.height,
          filename: input.filename,
          wasScaled: true,
          originalBytes,
          finalBytes: outputBuffer.length,
          selectedQuality: attempt.quality,
        }
      }
    }

    if (bestCandidate) {
      logger.warn(
        {
          originalBytes,
          finalBytes: bestCandidate.buffer.length,
          width: bestCandidate.width,
          height: bestCandidate.height,
          quality: bestCandidate.quality,
        },
        'Image autoscaling could not reduce under 5MB; attaching best-effort JPEG candidate',
      )

      return {
        base64: bestCandidate.buffer.toString('base64'),
        mime: 'image/jpeg',
        width: bestCandidate.width,
        height: bestCandidate.height,
        filename: input.filename,
        wasScaled: true,
        originalBytes,
        finalBytes: bestCandidate.buffer.length,
        selectedQuality: bestCandidate.quality,
      }
    }

    logger.warn({ originalBytes }, 'Image autoscaling unavailable or failed; attaching original image')
    return {
      ...input,
      wasScaled: false,
      originalBytes,
      finalBytes: originalBytes,
    }
  } catch (error) {
    logger.warn({ error, originalBytes }, 'Image autoscaling failed unexpectedly; attaching original image')
    return {
      ...input,
      wasScaled: false,
      originalBytes,
      finalBytes: originalBytes,
    }
  }
}