import type { ImagePart, ImageMediaType, ToolResultPart } from '@magnitudedev/ai'
import { ContentBuilder } from '../content'

export function isImageValue(value: unknown): value is Record<string, unknown> & { mediaType: string } {
  if (typeof value !== 'object' || value === null) return false
  const o = value as Record<string, unknown>
  return (
    typeof o.mediaType === 'string' &&
    (typeof o.data === 'string' || typeof o.base64 === 'string')
  )
}

export function toImagePart(value: Record<string, unknown> & { mediaType: string }): ImagePart {
  const data = typeof value.data === 'string' ? value.data : value.base64 as string
  const w = value.width, h = value.height
  const dimensions = typeof w === 'number' && typeof h === 'number' ? { width: w, height: h } : undefined
  return {
    _tag: 'ImagePart' as const,
    data,
    mediaType: value.mediaType as ImageMediaType,
    ...(dimensions ? { dimensions } : {}),
  }
}

export function isScalar(value: unknown): value is string | number | boolean | null {
  return value === null || typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
}

// --- Internal rendering functions ---

function renderScalar(value: string | number | boolean | null): string {
  return String(value)
}

function renderValueInto(builder: ContentBuilder, value: unknown): void {
  if (value === undefined) return
  if (isScalar(value)) {
    builder.pushText(renderScalar(value))
    return
  }
  if (isImageValue(value)) {
    builder.pushPart(toImagePart(value))
    return
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      if (i > 0) builder.pushText('\n')
      renderFieldInto(builder, String(i), value[i])
    }
    return
  }
  if (typeof value === 'object' && value !== null) {
    const entries = Object.entries(value).filter(([, v]) => v !== undefined)
    for (let i = 0; i < entries.length; i++) {
      if (i > 0) builder.pushText('\n')
      renderFieldInto(builder, entries[i][0], entries[i][1])
    }
    return
  }
  builder.pushText(String(value))
}

function renderFieldInto(builder: ContentBuilder, name: string, value: unknown): void {
  if (isScalar(value)) {
    const raw = renderScalar(value)
    if (!raw.includes('\n')) {
      builder.pushText(`<${name}>${raw}</${name}>`)
    } else {
      builder.pushText(`<${name}>\n${raw}\n</${name}>`)
    }
    return
  }
  builder.pushText(`<${name}>\n`)
  renderValueInto(builder, value)
  builder.pushText(`\n</${name}>`)
}

// --- Exported rendering functions ---

export function renderToolOutput(output: unknown): readonly ToolResultPart[] {
  const builder = new ContentBuilder()
  renderValueInto(builder, output)
  return builder.build()
}

export function renderTagged(tag: string, value: unknown): readonly ToolResultPart[] {
  const builder = new ContentBuilder()
  if (isScalar(value)) {
    const raw = renderScalar(value)
    if (!raw.includes('\n')) {
      builder.pushText(`<${tag}>${raw}</${tag}>`)
    } else {
      builder.pushText(`<${tag}>\n${raw}\n</${tag}>`)
    }
  } else {
    builder.pushText(`<${tag}>\n`)
    renderValueInto(builder, value)
    builder.pushText(`\n</${tag}>`)
  }
  return builder.build()
}
