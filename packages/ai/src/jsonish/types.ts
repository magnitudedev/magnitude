export type CompletionState = "complete" | "incomplete"

export type ParsedValue =
  | ParsedString
  | ParsedNumber
  | ParsedBoolean
  | ParsedNull
  | ParsedObject
  | ParsedArray

export interface ParsedString {
  readonly _tag: "string"
  readonly value: string
  readonly state: CompletionState
}

export interface ParsedNumber {
  readonly _tag: "number"
  readonly value: string
  readonly state: CompletionState
}

export interface ParsedBoolean {
  readonly _tag: "boolean"
  readonly value: boolean
  readonly state: "complete"
}

export interface ParsedNull {
  readonly _tag: "null"
  readonly state: "complete"
}

export interface ParsedObject {
  readonly _tag: "object"
  readonly entries: Array<[string, ParsedValue]>
  readonly state: CompletionState
}

export interface ParsedArray {
  readonly _tag: "array"
  readonly items: ParsedValue[]
  readonly state: CompletionState
}

export type JsonCollection =
  | ObjectCollection
  | ArrayCollection
  | QuotedStringCollection
  | UnquotedStringCollection

export interface ObjectCollection {
  readonly _tag: "object"
  keys: string[]
  values: ParsedValue[]
  state: CompletionState
}

export interface ArrayCollection {
  readonly _tag: "array"
  items: ParsedValue[]
  state: CompletionState
}

export interface QuotedStringCollection {
  readonly _tag: "quotedString"
  content: string
  state: CompletionState
  trailingBackslashes: number
  unescapedQuoteCount: number
}

export interface UnquotedStringCollection {
  readonly _tag: "unquotedString"
  content: string
  state: CompletionState
}

export type CloseStringResult =
  | { readonly _tag: "close"; readonly charsConsumed: number; readonly completion: CompletionState }
  | { readonly _tag: "continue" }

export type Pos =
  | { readonly _tag: "inNothing" }
  | { readonly _tag: "unknown" }
  | { readonly _tag: "inObjectKey" }
  | { readonly _tag: "inObjectValue" }
  | { readonly _tag: "inArray" }

export interface StreamingJsonParser {
  push(chunk: string): void
  end(): void
  readonly partial: ParsedValue | undefined
  readonly done: boolean
  readonly currentPath: readonly string[]
}
