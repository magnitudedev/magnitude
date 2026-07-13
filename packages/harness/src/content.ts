import type { TextPart, ImagePart, ToolResultPart } from '@magnitudedev/ai'

/** Builder for assembling content parts with text coalescing */
export class ContentBuilder {
  private parts: ToolResultPart[] = []

  pushText(text: string): void {
    if (!text) return
    const last = this.parts[this.parts.length - 1]
    if (last?._tag === 'TextPart') {
      this.parts[this.parts.length - 1] = { _tag: 'TextPart', text: last.text + text }
    } else {
      this.parts.push({ _tag: 'TextPart', text })
    }
  }

  pushPart(part: ToolResultPart): void {
    if (part._tag === 'TextPart') this.pushText(part.text)
    else this.parts.push(part)
  }

  pushParts(parts: readonly ToolResultPart[]): void {
    for (const part of parts) this.pushPart(part)
  }

  hasContent(): boolean { return this.parts.length > 0 }
  build(): ToolResultPart[] { return [...this.parts] }
}
