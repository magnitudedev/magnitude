import type { ContentPart, ImageMediaType } from '@magnitudedev/tools'

export type { ContentPart, ImageMediaType }

/** Wrap a plain string as ContentPart[] */
export function textParts(s: string): ContentPart[] {
  return [{ type: 'text', text: s }]
}

/** Create an image ContentPart */
export function imagePart(base64: string, mediaType: ImageMediaType, width: number, height: number): ContentPart {
  return { type: 'image', base64, mediaType, width, height }
}

/** Extract all text from ContentPart[], joining with newline */
export function textOf(parts: readonly ContentPart[] | null | undefined): string {
  if (!parts || !Array.isArray(parts)) return ''
  return parts.filter(p => p.type === 'text').map(p => p.text).join('\n')
}

/** Check if any part is an image */
export function hasImages(parts: ContentPart[]): boolean {
  return parts.some(p => p.type === 'image')
}

/** Apply a transform to text content while preserving image parts */
export function wrapTextParts(parts: ContentPart[], transform: (text: string) => string): ContentPart[] {
  const allText = parts.filter(p => p.type === 'text').map(p => p.text).join('\n')
  const wrapped = transform(allText)
  return [
    { type: 'text', text: wrapped },
    ...parts.filter(p => p.type !== 'text')
  ]
}

/** Migration helper: convert old string content to ContentPart[] */
export function migrateContent(content: string | ContentPart[]): ContentPart[] {
  if (typeof content === 'string') return textParts(content)
  return content
}
