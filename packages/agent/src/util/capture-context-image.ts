import { mkdir, writeFile } from 'node:fs/promises'
import { basename, join } from 'node:path'
import { createId } from '@magnitudedev/generate-id'
import type { ImageMediaType } from '@magnitudedev/ai'
import type { ContextImagePart } from '../content'
import { Data, Effect, Option } from 'effect'
import { readImageFileForModel } from './read-image-file'

export class ContextImageCaptureError extends Data.TaggedError('ContextImageCaptureError')<{
  readonly message: string
  readonly cause: unknown
}> {}

function captureError(cause: unknown): ContextImageCaptureError {
  return new ContextImageCaptureError({
    message: cause instanceof Error ? cause.message : 'Failed to capture image',
    cause,
  })
}

const EXTENSION_BY_MEDIA_TYPE: Record<ImageMediaType, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'image/gif': 'gif',
}

function sanitizeStem(value: string): string {
  const withoutExtension = value.replace(/\.[^.]+$/, '')
  const sanitized = withoutExtension.replace(/[^a-zA-Z0-9._-]/g, '_').replace(/^\.+/, '')
  return sanitized.slice(0, 80) || 'image'
}

export interface CaptureContextImageFromFileInput {
  readonly absolutePath: string
  readonly logicalPath: string
  readonly name?: string
}

/** Capture a bounded model-ready rendition while preserving an optimistic path. */
export function captureContextImageFromFile(
  input: CaptureContextImageFromFileInput,
): Effect.Effect<ContextImagePart, ContextImageCaptureError> {
  return Effect.tryPromise({
    try: async () => {
      const image = await readImageFileForModel(input.absolutePath)
      const normalizedBytes = Buffer.from(image.base64, 'base64')
      const captured: ContextImagePart = {
        _tag: 'ContextImage',
        data: image.base64,
        mediaType: image.mediaType,
        dimensions: { width: image.width, height: image.height },
        path: input.logicalPath,
        name: Option.some(input.name ?? basename(input.logicalPath)),
        byteSize: Option.some(normalizedBytes.byteLength),
      }
      return captured
    },
    catch: captureError,
  })
}

export interface CaptureContextImageInlineInput {
  readonly base64: string
  readonly mediaType: ImageMediaType
  readonly scratchpadPath: string
  readonly name?: string
  readonly preferredFilename?: string
}

/** Materialize inline bytes under $M/images, then capture the common semantic shape. */
export function captureContextImageInline(
  input: CaptureContextImageInlineInput,
): Effect.Effect<ContextImagePart, ContextImageCaptureError> {
  return Effect.gen(function* () {
    const bytes = Buffer.from(input.base64, 'base64')
    const extension = EXTENSION_BY_MEDIA_TYPE[input.mediaType]
    const stem = sanitizeStem(input.preferredFilename ?? input.name ?? 'image')
    const filename = `${stem}-${createId().slice(0, 12)}.${extension}`
    const imagesDirectory = join(input.scratchpadPath, 'images')
    const absolutePath = join(imagesDirectory, filename)
    yield* Effect.tryPromise({
      try: async () => {
        await mkdir(imagesDirectory, { recursive: true })
        await writeFile(absolutePath, bytes, { flag: 'wx' })
      },
      catch: captureError,
    })
    return yield* captureContextImageFromFile({
      absolutePath,
      logicalPath: `$M/images/${filename}`,
      name: input.name ?? input.preferredFilename ?? filename,
    })
  })
}
