import type { StreamingLeaf, StreamingPartial } from '@magnitudedev/tools'
import type { ToolCallEvent } from './types'

interface FieldMapping {
  segments: string[]
}

export interface StreamingAccumulatorConfig {
  attrs: Map<string, FieldMapping>
  bodyField: FieldMapping | null
  childFields: Map<string, FieldMapping>
}

export class StreamingAccumulator<TInput> {
  private _shape: Record<string, any> = {}

  constructor(private readonly config: StreamingAccumulatorConfig) {}

  ingest(event: ToolCallEvent): void {
    switch (event._tag) {
      case 'ToolInputStarted':
        this._shape = {}
        break

      case 'ToolInputFieldValue': {
        const mapping = this.config.attrs.get(String(event.field))
        if (mapping) {
          this.setPath(mapping.segments, { isFinal: true, value: event.value } satisfies StreamingLeaf<unknown>)
        }
        break
      }

      case 'ToolInputBodyChunk': {
        const path = event.path
        if (!path || path.length <= 1) {
          if (this.config.bodyField) {
            const segs = this.config.bodyField.segments
            const current = this.getPath(segs) as StreamingLeaf<unknown> | undefined
            const prior = current?.value ?? ''
            this.setPath(segs, { isFinal: false, value: `${prior}${event.text}` } satisfies StreamingLeaf<unknown>)
          }
        } else {
          const mapping = this.config.childFields.get(String(path[0]))
          if (mapping) {
            const current = this.getPath(mapping.segments) as StreamingLeaf<unknown> | undefined
            const prior = current?.value ?? ''
            this.setPath(mapping.segments, { isFinal: false, value: `${prior}${event.text}` } satisfies StreamingLeaf<unknown>)
          }
        }
        break
      }

      case 'ToolInputChildComplete': {
        const mapping = this.config.childFields.get(String(event.field))
        if (mapping) {
          const current = this.getPath(mapping.segments) as StreamingLeaf<unknown> | undefined
          if (current) {
            this.setPath(mapping.segments, { isFinal: true, value: current.value } satisfies StreamingLeaf<unknown>)
          }
        }
        break
      }

      case 'ToolInputReady': {
        this.markAllLeavesFinal(this._shape)
        break
      }
    }
  }

  get current(): StreamingPartial<TInput> {
    return { ...this._shape } as StreamingPartial<TInput>
  }

  reset(): void {
    this._shape = {}
  }

  private setPath(segments: string[], value: any): void {
    if (segments.length === 1) {
      this._shape[segments[0]] = value
      return
    }
    let obj = this._shape
    for (let i = 0; i < segments.length - 1; i++) {
      if (!(segments[i] in obj) || typeof obj[segments[i]] !== 'object') {
        obj[segments[i]] = {}
      }
      obj = obj[segments[i]]
    }
    obj[segments[segments.length - 1]] = value
  }

  private getPath(segments: string[]): any {
    if (segments.length === 1) return this._shape[segments[0]]
    let obj: any = this._shape
    for (const seg of segments) {
      if (obj == null || typeof obj !== 'object') return undefined
      obj = obj[seg]
    }
    return obj
  }

  private markAllLeavesFinal(node: unknown): void {
    if (!node || typeof node !== 'object') return
    const record = node as Record<string, unknown>
    for (const [key, value] of Object.entries(record)) {
      if (
        value &&
        typeof value === 'object' &&
        'isFinal' in value &&
        'value' in value
      ) {
        const leaf = value as StreamingLeaf<unknown>
        if (!leaf.isFinal) {
          record[key] = { isFinal: true, value: leaf.value } satisfies StreamingLeaf<unknown>
        }
      } else {
        this.markAllLeavesFinal(value)
      }
    }
  }
}
