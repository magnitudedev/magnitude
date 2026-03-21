import type { ChildAcc, StreamingInput } from '@magnitudedev/tools'
import type { ToolCallEvent } from './types'

export class SchemaAccumulator<
  TAccFields extends Record<string, string | undefined> = Record<string, string | undefined>,
  TAccChildren extends Record<string, ChildAcc[]> = Record<string, ChildAcc[]>
> {
  private _fields: Record<string, string> = {}
  private _body = ''
  private _children: Record<string, ChildAcc[]> = {}

  constructor() {}

  /**
   * Ingest a raw ToolCallEvent from the xml-act parser.
   * Updates the accumulated streaming input accordingly.
   */
  ingest(event: ToolCallEvent): void {
    switch (event._tag) {
      case 'ToolInputStarted': {
        this.reset()
        break
      }
      case 'ToolInputFieldValue': {
        this._fields[event.field] = String(event.value)
        break
      }
      case 'ToolInputBodyChunk': {
        const path = event.path
        if (!path || path.length === 0 || path.length === 1) {
          this._body += event.text
          break
        }

        const childTag = path[0]
        const childIndex = Number(path[1])
        if (
          typeof childTag === 'string' &&
          Number.isFinite(childIndex) &&
          this._children[childTag]?.[childIndex]
        ) {
          this._children[childTag][childIndex].body += event.text
        }
        break
      }
      case 'ToolInputChildStarted': {
        const tag = String(event.field)
        if (!this._children[tag]) {
          this._children[tag] = []
        }

        const attrs: Record<string, string> = {}
        const rawAttrs = event.attributes as Record<string, unknown> | undefined
        if (rawAttrs) {
          for (const [key, value] of Object.entries(rawAttrs)) {
            attrs[key] = String(value)
          }
        }

        const nextIndex = event.index
        this._children[tag][nextIndex] = {
          body: '',
          complete: false,
          attrs,
        }
        break
      }
      case 'ToolInputChildComplete': {
        const tag = String(event.field)
        const index = event.index
        if (this._children[tag]?.[index]) {
          this._children[tag][index].complete = true
        }
        break
      }
      default:
        break
    }
  }

  /** Current accumulated snapshot */
  get current(): StreamingInput<TAccFields, TAccChildren> {
    return {
      fields: { ...this._fields } as Partial<TAccFields>,
      body: this._body,
      children: Object.fromEntries(
        Object.entries(this._children).map(([key, value]) => [
          key,
          value.map((child) => ({ ...child, attrs: { ...child.attrs } })),
        ]),
      ) as StreamingInput<TAccFields, TAccChildren>['children'],
    }
  }

  /** Reset for reuse */
  reset(): void {
    this._fields = {}
    this._body = ''
    this._children = {}
  }
}
