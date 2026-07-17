import { Effect, Option } from 'effect'
import { describe, expect, it } from 'vitest'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { renderContextImageAnchor, renderContextParts, type ContextImagePart } from '../src/content'
import { captureContextImageInline } from '../src/util/capture-context-image'

const image: ContextImagePart = {
  _tag: 'ContextImage', data: 'YWJj', mediaType: 'image/png',
  dimensions: { width: 12, height: 8 },
  path: '$M/images/example.png', name: Option.some('example.png'),
  byteSize: Option.some(3),
}

describe('context image rendering', () => {
  it('keeps the same stable anchor with and without pixels', () => {
    const anchor = renderContextImageAnchor(image)
    const nonVision = renderContextParts([image], { includeImageData: false })
    const vision = renderContextParts([image], { includeImageData: true })
    expect(nonVision).toEqual([{ _tag: 'TextPart', text: anchor }])
    expect(vision[0]).toEqual(nonVision[0])
    expect(vision[1]).toMatchObject({ _tag: 'ImagePart', data: 'YWJj', mediaType: 'image/png' })
  })

  it('materializes inline bytes before returning canonical image content', async () => {
    const scratchpadPath = await mkdtemp(join(tmpdir(), 'magnitude-context-image-'))
    try {
      const captured = await Effect.runPromise(captureContextImageInline({
        base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
        mediaType: 'image/png',
        scratchpadPath,
        name: 'pixel.png',
      }))
      const file = await readFile(join(scratchpadPath, captured.path.replace(/^\$M\//, '')))

      expect(file.byteLength).toBeGreaterThan(0)
      expect(captured.data.length).toBeGreaterThan(0)
      expect(captured.path).toMatch(/^\$M\/images\/pixel-/)
    } finally {
      await rm(scratchpadPath, { recursive: true, force: true })
    }
  })
})
