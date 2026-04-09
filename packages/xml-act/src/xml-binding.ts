import type { FieldPath, ToolBinding } from '@magnitudedev/tools'
import { StreamingAccumulator, type StreamingAccumulatorConfig } from './streaming-accumulator'
import type { XmlTagBinding } from './types'

type ArrayElement<T> = T extends ReadonlyArray<infer E> ? E : never
type ArrayFieldKeys<T> = {
  [K in keyof T]-?: T[K] extends ReadonlyArray<unknown> ? K : never
}[keyof T] & string
type ArrayFieldElement<T, K extends keyof T> = T[K] extends ReadonlyArray<infer E> ? E : never
type XmlOutputChildrenBinding<TOutput> = readonly {
  [K in ArrayFieldKeys<TOutput>]: {
    field: K
    tag?: string
    attributes?: readonly { attr: string; field: keyof ArrayFieldElement<TOutput, K> & string }[]
    body?: keyof ArrayFieldElement<TOutput, K> & string
  }
}[ArrayFieldKeys<TOutput>][]

// The mapping config for XML input bindings
export interface XmlInputMappingConfig<TInput> {
  attributes?: readonly { attr: string; field: FieldPath<TInput> }[]
  body?: FieldPath<TInput>
  selfClosing?: boolean
  childTags?: readonly { tag: string; field: FieldPath<TInput> }[]
  children?: readonly {
    field: keyof TInput & string
    tag?: string
    attributes?: readonly { attr: string; field: string }[]
    body?: string
  }[]
  childRecord?: {
    field: keyof TInput & string
    tag: string
    keyAttr: string
  }
}

export interface XmlOutputBinding<TOutput> {
  tag?: string
  attributes?: readonly { attr: string; field: FieldPath<TOutput> }[]
  body?: FieldPath<TOutput>
  childTags?: readonly { tag: string; field: FieldPath<TOutput> }[]
  children?: XmlOutputChildrenBinding<TOutput>
  childRecord?: {
    field: FieldPath<TOutput>
    tag: string
    keyAttr: string
  }
  items?: TOutput extends ReadonlyArray<unknown> ? {
    tag?: string
    attributes?: readonly { attr: string; field: keyof ArrayElement<TOutput> & string }[]
    body?: keyof ArrayElement<TOutput> & string
  } : never
}

// Full mapping config including output
export interface XmlMappingConfig<TInput, TOutput = unknown> {
  group?: string
  tag?: string
  input: XmlInputMappingConfig<TInput>
  output?: XmlOutputBinding<TOutput>
}

export interface XmlBindingResult<TInput, TOutput, TMapping>
  extends ToolBinding<TInput> {
  readonly tool: { name: string; group?: string }
  readonly config: TMapping

  /**
   * Convert to the runtime XmlTagBinding format that xml-act's
   * buildInput and dispatcher consume.
   */
  toXmlTagBinding(): XmlTagBinding

  /**
   * Convert to the runtime output binding format.
   */
  toXmlOutputBinding(): XmlOutputBinding<TOutput>

  createAccumulator(): StreamingAccumulator<TInput>
}

/**
 * Define an XML binding for a tool. The `field` values in the input mapping
 * are constrained to `keyof TInput` for compile-time safety.
 *
 * Use `as const` on the config to preserve literal types for streaming shape derivation.
 */
export function defineXmlBinding<
  TInput,
  TOutput,
  const TMapping extends XmlMappingConfig<TInput, TOutput>
>(
  tool: {
    readonly name: string
    readonly group?: string
    readonly inputSchema: { readonly Type: TInput }
    readonly outputSchema: { readonly Type: TOutput }
  },
  config: TMapping,
): XmlBindingResult<TInput, TOutput, TMapping> {
  return {
    tool: { name: tool.name, group: config.group ?? tool.group },
    config,
    toXmlTagBinding(): XmlTagBinding {
      const input = config.input
      return {
        tag: config.tag ?? tool.name,
        ...(input.attributes && {
          attributes: input.attributes.map((a) => ({ attr: a.attr, field: a.field })),
        }),
        ...(input.body && { body: input.body }),
        ...(input.selfClosing !== undefined && { selfClosing: input.selfClosing }),
        ...(input.childTags && {
          childTags: input.childTags.map((ct) => ({ tag: ct.tag, field: ct.field })),
        }),
        ...(input.children && {
          children: input.children.map((ch) => ({
            field: ch.field,
            tag: ch.tag,
            attributes: ch.attributes ? ch.attributes.map((a) => ({ attr: a.attr, field: a.field })) : undefined,
            body: ch.body,
          })),
        }),
        ...(input.childRecord && {
          childRecord: {
            field: input.childRecord.field,
            tag: input.childRecord.tag,
            keyAttr: input.childRecord.keyAttr,
          },
        }),
      }
    },
    toXmlOutputBinding() {
      const output: XmlOutputBinding<TOutput> = config.output ?? {}
      return output
    },
    createAccumulator() {
      const input = config.input

      const attrs: StreamingAccumulatorConfig['attrs'] = new Map()
      for (const a of input.attributes ?? []) {
        attrs.set(a.attr, { segments: a.field.split('.') })
      }

      const bodyField: StreamingAccumulatorConfig['bodyField'] = input.body
        ? { segments: input.body.split('.') }
        : null

      const childFields: StreamingAccumulatorConfig['childFields'] = new Map()
      for (const ct of input.childTags ?? []) {
        childFields.set(ct.field, { segments: ct.field.split('.') })
      }

      return new StreamingAccumulator<TInput>({ attrs, bodyField, childFields })
    },
  }
}
