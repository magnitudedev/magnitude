import type { TextPart, ImagePart, ImageMediaType } from '@magnitudedev/ai'

// Re-export ai types for convenience
export type { TextPart, ImagePart, ImageMediaType } from '@magnitudedev/ai'

/** The content part union — re-exported from ai */
export type { UserPart } from '@magnitudedev/ai'

/** Wrap a plain string as UserPart[] */
export function textParts(s: string): [TextPart] {
  return [{ _tag: 'TextPart', text: s }]
}

/** Extract all text from parts, joining with newline */
export function textOf(parts: readonly (TextPart | ImagePart)[] | null | undefined): string {
  if (!parts || !Array.isArray(parts)) return ''
  return parts.filter((p): p is TextPart => p._tag === 'TextPart').map(p => p.text).join('\n')
}

/** Check if any part is an image */
export function hasImages(parts: readonly (TextPart | ImagePart)[]): boolean {
  return parts.some(p => p._tag === 'ImagePart')
}

/** Apply a transform to text content while preserving image parts */
export function wrapTextParts(parts: readonly (TextPart | ImagePart)[], transform: (text: string) => string): (TextPart | ImagePart)[] {
  const allText = parts.filter((p): p is TextPart => p._tag === 'TextPart').map(p => p.text).join('\n')
  return [
    { _tag: 'TextPart', text: transform(allText) } satisfies TextPart,
    ...parts.filter((p): p is ImagePart => p._tag === 'ImagePart')
  ]
}

/** Builder for assembling content parts with text coalescing */
export { ContentBuilder } from '@magnitudedev/harness'
