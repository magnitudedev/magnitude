import { Option, Schema } from 'effect'
import type {
  ImageMediaType,
  ImagePart as AiImagePart,
  TextPart as AiTextPart,
  UserPart as AiUserPart,
} from '@magnitudedev/ai'

export const ContextTextPartSchema = Schema.TaggedStruct('ContextText', {
  text: Schema.String,
})
export type ContextTextPart = typeof ContextTextPartSchema.Type

/**
 * Durable agent-native image content.
 *
 * `data` is the durable pixel authority; `path` is the optimistic file-tool
 * affordance and may later move, change, or disappear.
 */
export const ContextImagePartSchema = Schema.TaggedStruct('ContextImage', {
  data: Schema.String,
  mediaType: Schema.Literal('image/png', 'image/jpeg', 'image/webp', 'image/gif'),
  dimensions: Schema.Struct({
    width: Schema.NonNegativeInt,
    height: Schema.NonNegativeInt,
  }),
  path: Schema.String,
  name: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  byteSize: Schema.optionalWith(Schema.NonNegativeInt, { as: 'Option', exact: true }),
})
export type ContextImagePart = typeof ContextImagePartSchema.Type

export const ContextPartSchema = Schema.Union(ContextTextPartSchema, ContextImagePartSchema)
export type ContextPart = typeof ContextPartSchema.Type

export const ContextImageResultSchema = Schema.TaggedStruct('ContextImageResult', {
  image: ContextImagePartSchema,
})
export type ContextImageResult = typeof ContextImageResultSchema.Type

export type { ImageMediaType }

/** Wrap a plain string as semantic ContextPart[]. */
export function textParts(s: string): [ContextTextPart] {
  return [{ _tag: 'ContextText', text: s }]
}

/** Extract all text from semantic parts, joining with newline. */
export function textOf(parts: readonly ContextPart[] | null | undefined): string {
  if (!parts || !Array.isArray(parts)) return ''
  return parts.filter((p): p is ContextTextPart => p._tag === 'ContextText').map(p => p.text).join('\n')
}

export function hasImages(parts: readonly ContextPart[]): boolean {
  return parts.some(p => p._tag === 'ContextImage')
}

/** Apply a transform to text content while preserving semantic image parts. */
export function wrapTextParts(parts: readonly ContextPart[], transform: (text: string) => string): ContextPart[] {
  const allText = parts.filter((p): p is ContextTextPart => p._tag === 'ContextText').map(p => p.text).join('\n')
  return [
    { _tag: 'ContextText', text: transform(allText) },
    ...parts.filter((p): p is ContextImagePart => p._tag === 'ContextImage'),
  ]
}

/** Stable model-independent textual anchor for an image. */
export function renderContextImageAnchor(image: ContextImagePart): string {
  const details = [
    image.path,
    image.mediaType,
    `${image.dimensions.width}x${image.dimensions.height}`,
  ]
  if (Option.isSome(image.byteSize)) details.push(`${image.byteSize.value} bytes`)
  return `[Image: ${details.join(' | ')}]`
}

export interface ContextRenderPolicy {
  readonly includeImageData: boolean
}

/** Convert durable semantic context into provider transport parts. */
export function renderContextParts(
  parts: readonly ContextPart[],
  policy: ContextRenderPolicy,
): AiUserPart[] {
  const rendered: AiUserPart[] = []
  for (const part of parts) {
    if (part._tag === 'ContextText') {
      rendered.push({ _tag: 'TextPart', text: part.text } satisfies AiTextPart)
      continue
    }
    rendered.push({ _tag: 'TextPart', text: renderContextImageAnchor(part) } satisfies AiTextPart)
    if (policy.includeImageData) {
      rendered.push({
        _tag: 'ImagePart',
        data: part.data,
        mediaType: part.mediaType,
        dimensions: part.dimensions,
      } satisfies AiImagePart)
    }
  }
  return rendered
}

export function isContextImagePart(value: unknown): value is ContextImagePart {
  return Schema.is(ContextImagePartSchema)(value)
}

/** Small accumulator for composing durable context without crossing into AI transport types. */
export class ContextBuilder {
  readonly #parts: ContextPart[] = []

  pushText(text: string): void {
    if (!text) return
    const last = this.#parts[this.#parts.length - 1]
    if (last?._tag === 'ContextText') {
      this.#parts[this.#parts.length - 1] = { _tag: 'ContextText', text: last.text + text }
    } else {
      this.#parts.push({ _tag: 'ContextText', text })
    }
  }

  pushPart(part: ContextPart): void {
    if (part._tag === 'ContextText') this.pushText(part.text)
    else this.#parts.push(part)
  }

  hasContent(): boolean {
    return this.#parts.length > 0
  }

  build(): ContextPart[] {
    return [...this.#parts]
  }
}
