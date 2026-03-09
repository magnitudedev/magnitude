import { $ } from 'bun'
import { logger } from '@magnitudedev/logger'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { extractImageDimensions } from './clipboard'

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

async function scaleWithSips(inputPath: string, outputPath: string, maxDimension: number, quality: number): Promise<boolean> {
  const result = await $`sips -s format jpeg -s formatOptions ${String(quality)} --resampleHeightWidthMax ${String(maxDimension)} ${inputPath} --out ${outputPath}`.nothrow().quiet()
  return result.exitCode === 0
}

async function scaleWithConvert(inputPath: string, outputPath: string, width: number, height: number, quality: number): Promise<boolean> {
  const result = await $`convert ${inputPath} -resize ${`${width}x${height}>`} -quality ${String(quality)} ${outputPath}`.nothrow().quiet()
  return result.exitCode === 0
}

function ffmpegQvFromJpegQuality(quality: number): number {
  return Math.max(2, Math.min(31, Math.round((100 - quality) / 3) + 2))
}

async function scaleWithFfmpeg(inputPath: string, outputPath: string, width: number, height: number, quality: number): Promise<boolean> {
  const qv = ffmpegQvFromJpegQuality(quality)
  const filter = `scale=${width}:${height}:force_original_aspect_ratio=decrease`
  const result = await $`ffmpeg -y -i ${inputPath} -vf ${filter} -q:v ${String(qv)} ${outputPath}`.nothrow().quiet()
  return result.exitCode === 0
}

function psEscapeSingleQuoted(value: string): string {
  return value.replace(/'/g, "''")
}

async function scaleWithPowerShellDotNet(
  inputPath: string,
  outputPath: string,
  width: number,
  height: number,
  quality: number,
): Promise<boolean> {
  const inEscaped = psEscapeSingleQuoted(inputPath)
  const outEscaped = psEscapeSingleQuoted(outputPath)
  const script = `
Add-Type -AssemblyName System.Drawing
$inputPath = '${inEscaped}'
$outputPath = '${outEscaped}'
$maxW = ${width}
$maxH = ${height}
$quality = ${quality}

$img = [System.Drawing.Image]::FromFile($inputPath)
try {
  $ratioW = $maxW / $img.Width
  $ratioH = $maxH / $img.Height
  $ratio = [Math]::Min(1.0, [Math]::Min($ratioW, $ratioH))
  $targetW = [Math]::Max(1, [int]([Math]::Floor($img.Width * $ratio)))
  $targetH = [Math]::Max(1, [int]([Math]::Floor($img.Height * $ratio)))

  $bmp = New-Object System.Drawing.Bitmap($targetW, $targetH)
  try {
    $graphics = [System.Drawing.Graphics]::FromImage($bmp)
    try {
      $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
      $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
      $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
      $graphics.DrawImage($img, 0, 0, $targetW, $targetH)
    } finally {
      $graphics.Dispose()
    }

    $encoder = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' } | Select-Object -First 1
    if (-not $encoder) { exit 1 }

    $encParams = New-Object System.Drawing.Imaging.EncoderParameters(1)
    $encParams.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, [long]$quality)
    $bmp.Save($outputPath, $encoder, $encParams)
  } finally {
    $bmp.Dispose()
  }
} finally {
  $img.Dispose()
}
`
  const result = await $`powershell.exe -NonInteractive -NoProfile -Command ${script}`.nothrow().quiet()
  return result.exitCode === 0
}

async function scaleForPlatform(inputPath: string, outputPath: string, width: number, height: number, quality: number): Promise<boolean> {
  if (process.platform === 'darwin') {
    return scaleWithSips(inputPath, outputPath, Math.max(width, height), quality)
  }

  if (process.platform === 'linux') {
    if (await scaleWithConvert(inputPath, outputPath, width, height, quality)) return true
    return scaleWithFfmpeg(inputPath, outputPath, width, height, quality)
  }

  if (process.platform === 'win32') {
    return scaleWithPowerShellDotNet(inputPath, outputPath, width, height, quality)
  }

  return false
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

  const tempDir = await mkdtemp(path.join(tmpdir(), 'magnitude-image-scale-'))
  const inputPath = path.join(tempDir, 'input.bin')

  try {
    await writeFile(inputPath, Buffer.from(input.base64, 'base64'))

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
      const outputPath = path.join(tempDir, `out-${attempt.width}x${attempt.height}-q${attempt.quality}.jpg`)
      const ok = await scaleForPlatform(inputPath, outputPath, attempt.width, attempt.height, attempt.quality)
      if (!ok) continue

      let outputBuffer: Buffer
      try {
        outputBuffer = await readFile(outputPath)
      } catch {
        continue
      }

      if (outputBuffer.length === 0) continue

      const dims = extractImageDimensions(outputBuffer)
      if (!dims) continue

      const candidate = { buffer: outputBuffer, width: dims.width, height: dims.height, quality: attempt.quality }

      if (!bestCandidate || candidate.buffer.length < bestCandidate.buffer.length) {
        bestCandidate = candidate
      }

      if (outputBuffer.length <= MAX_BASE64_BYTES) {
        return {
          base64: outputBuffer.toString('base64'),
          mime: 'image/jpeg',
          width: dims.width,
          height: dims.height,
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
  } finally {
    await rm(tempDir, { recursive: true, force: true }).catch(() => {})
  }
}