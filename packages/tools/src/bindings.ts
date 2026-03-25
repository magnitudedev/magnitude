/**
 * @magnitudedev/tools — Tool Bindings
 *
 * Strategy-specific injection metadata for tools.
 * Declares how a tool should be presented to each LLM strategy
 * (OpenAI function calling, codegen/JS-ACT, XML-ACT).
 */

/**
 * Accumulated state of a child XML tag during streaming.
 * Tracks the tag's body text, completion status, and attributes.
 */
export type ChildAcc = {
  body: string;
  complete: boolean;
  attrs: Record<string, string>;
};

// =============================================================================
// Utility Types
// =============================================================================

/** Extract string field names from an input type. Falls back to string when T is unknown.
 *  Distributes over unions so that A | B yields keys from both variants. */
export type InputFields<T> = [T] extends [never]
  ? string
  : T extends unknown
    ? ([Extract<keyof T, string>] extends [never] ? string : Extract<keyof T, string>)
    : never

/** Extract fields of T whose values are arrays. Distributes over unions. */
export type ArrayFields<T> = T extends unknown ? {
  [K in keyof T]: T[K] extends ReadonlyArray<unknown> ? K : never
}[keyof T] & string : never

/** Extract element type from an array field */
export type ArrayElement<T, K extends keyof T> =
  T[K] extends ReadonlyArray<infer E> ? E : never

/** Extract keys of T whose values are plain objects (not arrays, not primitives).
 *  Handles optional fields by stripping undefined before checking.
 *  Distributes over unions. */
type ObjectFields<T> = T extends unknown ? {
  [K in keyof T]-?: NonNullable<T[K]> extends ReadonlyArray<unknown>
    ? never
    : NonNullable<T[K]> extends Record<string, unknown>
      ? K
      : never
}[keyof T] & string : never

/** Extract keys of T whose values are Record<string, string> (key-value maps).
 *  Distributes over unions. */
export type RecordFields<T> = T extends unknown ? {
  [K in keyof T]-?: NonNullable<T[K]> extends Record<string, string> ? K : never
}[keyof T] & string : never

/** Field paths for XML bindings.
 *  Top-level fields: 'name'. Nested fields: 'options.type'.
 *  Falls back to string when T is unknown.
 *  Distributes over unions so A | B yields paths from both variants. */
export type FieldPath<T> = InputFields<T> | (T extends unknown ? {
  [K in ObjectFields<T>]: `${K}.${Extract<keyof NonNullable<T[K]>, string>}`
}[ObjectFields<T>] : never)

/** Backward-compatible alias for childTag bindings. */
export type ChildTagPath<T> = FieldPath<T>

export interface XmlAttrBinding<T> {
  readonly field: FieldPath<T>
  readonly attr: string
}

// =============================================================================
// OpenAI Custom Tool Format (CFG constraint)
// =============================================================================

/** Format constraint for OpenAI custom tools.
 *  Constrains model output via LLGuidance — either a Lark CFG or a regex. */
export interface CustomToolFormat {
  readonly type: 'grammar'
  readonly syntax: 'lark' | 'regex'
  /** The grammar definition string (Lark CFG or regex pattern). */
  readonly definition: string
}

// =============================================================================
// OpenAI Bindings
// =============================================================================

export type OpenAIBinding<T> =
  | {
      /** Standard JSON Schema function calling (default). */
      readonly type: 'function'
      /** Override the function name sent to OpenAI. Must match ^[a-zA-Z0-9_-]+$. */
      readonly name?: string
    }
  | {
      /** Freeform text payload — model sends raw text instead of structured JSON.
       *  Maps to OpenAI's { type: "custom" } tool definition.
       *  Output is constrained by a CFG (Lark or regex) via LLGuidance. */
      readonly type: 'custom'
      /** Grammar format constraining the model's freeform output.
       *  Maps to the `format` field on OpenAI custom tool definitions. */
      readonly format: CustomToolFormat
      /** Which input schema field receives the raw text from the model. */
      readonly inputField: InputFields<T>
      /** Custom description for freeform mode. Falls back to tool.description if absent. */
      readonly description?: string
    }
  | {
      /** Strategy handles this tool natively — not exposed as a tool to the model.
       *  The strategy maps it to a native mechanism (text output, reasoning tokens, etc.). */
      readonly type: 'native'
      /** Which native mechanism to use. */
      readonly mechanism: 'text-output' | 'reasoning'
    }
  | {
      /** Don't expose to the model at all. */
      readonly type: 'omit'
    }

// =============================================================================
// Codegen Bindings (JS-ACT)
// =============================================================================

export type CodegenBinding =
  | {
      /** Normal callable tool — ETS generates TypeScript documentation (default). */
      readonly type: 'callable'
      /** Whether to show in a `declare namespace` block (default: true for grouped tools). */
      readonly showInNamespace?: boolean
      /** Whether to extract common entity types (default: true). */
      readonly extractCommon?: boolean
    }
  | {
      /** Don't expose to the model. */
      readonly type: 'omit'
    }

// =============================================================================
// XML Bindings (XML-ACT)
// =============================================================================

/**
 * Base shape of an XML child binding (array field → repeated child tags).
 * Always usable — no generics, plain strings.
 */
export interface XmlChildBinding {
  readonly field: string
  readonly tag?: string
  readonly attributes?: readonly { readonly field: string; readonly attr: string }[]
  readonly body?: string
}

/**
 * Base shape of a childTag binding (scalar field → named child element).
 * `field` is a path: top-level ('name') or dotted ('options.type').
 * `tag` is the XML element name the model writes.
 */
export interface XmlChildTagBinding<T = unknown> {
  readonly field: FieldPath<T>
  readonly tag: string
}

/**
 * Strongly-typed child binding for array fields.
 * Distributes K across all array fields of T, so each child binding has
 * attributes/body typed against the element type of that specific array field.
 * Falls back to XmlChildBinding when T is erased (ArrayFields<T> = never).
 */
export type XmlArrayChildBinding<T> = [ArrayFields<T>] extends [never]
  ? XmlChildBinding
  : {
      [K in ArrayFields<T>]: Omit<XmlChildBinding, 'field' | 'attributes' | 'body'> & {
        readonly field: K
        readonly attributes?: ReadonlyArray<{
          readonly field: InputFields<ArrayElement<T, K>>
          readonly attr: string
        }>
        readonly body?: InputFields<ArrayElement<T, K>>
      }
    }[ArrayFields<T>]

/**
 * Item binding for direct array outputs (when output is array<E>).
 * Specifies how each array element is serialized to an XML item element.
 */
export type XmlItemBinding<E> = {
  readonly tag: string
  readonly attributes?: ReadonlyArray<{ readonly attr: string; readonly field: keyof E & string }>
  readonly body?: keyof E & string
}

export type XmlBinding<T> = {
  /** Map tool to an XML tag with typed field assignments. */
  readonly type: 'tag'
  /** Override the XML tag name. Defaults to tool name. */
  readonly tag?: string
  /** Input fields mapped to XML attributes. */
  readonly attributes?: ReadonlyArray<XmlAttrBinding<T>>
  /** Input field mapped to inner text content. */
  readonly body?: FieldPath<T>
  /** Whether the tag is self-closing when no body/children are present. */
  readonly selfClosing?: boolean
  /** Scalar fields rendered as child tags: <tag>value</tag>.
   *  Each entry specifies a field path and optional XML tag name override. */
  readonly childTags?: ReadonlyArray<XmlChildTagBinding<T>>
  /** Array fields rendered as repeated child tags with their own structure. */
  readonly children?: ReadonlyArray<XmlArrayChildBinding<T>>
  /** Record field rendered as repeated child elements with key attr + body value. */
  readonly childRecord?: {
    /** Which record field this child record maps to. */
    readonly field: [RecordFields<T>] extends [never] ? string : RecordFields<T>
    /** The child tag name */
    readonly tag: string
    /** Which attribute serves as the record key */
    readonly keyAttr: string
  }
  /** Direct array output binding (when output type is array<E>).
   *  Specifies how each array element is serialized as an item element. */
  readonly items?: [T] extends [ReadonlyArray<infer E>]
    ? XmlItemBinding<E>
    : [unknown] extends [T] ? XmlItemBinding<Record<string, unknown>> : never
}

// =============================================================================
// Combined Bindings
// =============================================================================

export interface ToolBindings<TInput, TOutput = unknown> {
  readonly xmlInput?: XmlBinding<TInput>
  readonly xmlOutput?: XmlBinding<TOutput>
}

// Streaming shape derivation from XML binding mapping
export type AttrNames<TMapping> =
  TMapping extends { attributes: readonly (infer A)[] }
    ? A extends { attr: infer N extends string } ? N : never
    : never;

export type ChildTagNames<TMapping> =
  TMapping extends { childTags: readonly (infer C)[] }
    ? C extends { tag: infer N extends string } ? N : never
    : never;

export type DeriveStreamingShape<TMapping> = {
  fields: { [K in AttrNames<TMapping>]?: string };
  body: TMapping extends { body: string } ? string : '';
  children: (
    { [K in ChildTagNames<TMapping>]?: ChildAcc[] }
  ) & (
    TMapping extends { childRecord: { tag: infer N extends string } }
      ? { [K in N]?: ChildAcc[] }
      : {}
  );
};
