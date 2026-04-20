
/**
 * JSONish streaming parser types
 * 
 * Ported from BAML's jsonish parser - handles incremental JSON parsing
 * with support for incomplete/malformed JSON common in LLM outputs.
 */

export type CompletionState = 'complete' | 'incomplete';

/**
 * Discriminated union of parsed JSON values.
 * Every non-primitive carries a CompletionState to track whether the
 * value was properly terminated in the stream.
 */
export type ParsedValue =
  | ParsedString
  | ParsedNumber
  | ParsedBoolean
  | ParsedNull
  | ParsedObject
  | ParsedArray;

export interface ParsedString {
  readonly _tag: 'string';
  readonly value: string;
  readonly state: CompletionState;
}

export interface ParsedNumber {
  readonly _tag: 'number';
  /** Stored as string because incomplete numbers like "12" aren't valid JS numbers */
  readonly value: string;
  readonly state: CompletionState;
}

export interface ParsedBoolean {
  readonly _tag: 'boolean';
  readonly value: boolean;
  /** Always 'complete' once recognized */
  readonly state: 'complete';
}

export interface ParsedNull {
  readonly _tag: 'null';
  /** Always 'complete' once recognized */
  readonly state: 'complete';
}

export interface ParsedObject {
  readonly _tag: 'object';
  /** Entries as [key, value] tuples to preserve order and handle incomplete keys */
  readonly entries: Array<[string, ParsedValue]>;
  readonly state: CompletionState;
}

export interface ParsedArray {
  readonly _tag: 'array';
  readonly items: ParsedValue[];
  readonly state: CompletionState;
}

/**
 * Collection types for the parser state machine.
 * These represent values currently being built on the stack.
 */
export type JsonCollection =
  | ObjectCollection
  | ArrayCollection
  | QuotedStringCollection
  | UnquotedStringCollection;

export interface ObjectCollection {
  readonly _tag: 'object';
  /** Keys that have been parsed (may be more than values if mid-key) */
  keys: string[];
  /** Values that have been parsed */
  values: ParsedValue[];
  state: CompletionState;
}

export interface ArrayCollection {
  readonly _tag: 'array';
  items: ParsedValue[];
  state: CompletionState;
}

export interface QuotedStringCollection {
  readonly _tag: 'quotedString';
  content: string;
  state: CompletionState;
  /** 
   * Incremental quote tracking for O(1) escape detection.
   * Count of consecutive backslashes at end of content.
   */
  trailingBackslashes: number;
  /**
   * Count of unescaped quotes (quotes preceded by even number of backslashes).
   * Used to determine if a quote should close the string.
   */
  unescapedQuoteCount: number;
}

export interface UnquotedStringCollection {
  readonly _tag: 'unquotedString';
  content: string;
  state: CompletionState;
}

/**
 * Result of checking whether an unquoted string should close.
 */
export type CloseStringResult =
  | { readonly _tag: 'close'; readonly charsConsumed: number; readonly completion: CompletionState }
  | { readonly _tag: 'continue' };

/**
 * Position context for determining which delimiters terminate an unquoted string.
 */
export type Pos =
  | { readonly _tag: 'inNothing' }      // Top level
  | { readonly _tag: 'unknown' }          // Unknown context
  | { readonly _tag: 'inObjectKey' }      // Inside object, reading a key
  | { readonly _tag: 'inObjectValue' }    // Inside object, reading a value  
  | { readonly _tag: 'inArray' };        // Inside array

/**
 * Streaming JSON parser interface.
 */
export interface StreamingJsonParser {
  /**
   * Push a chunk of text into the parser.
   */
  push(chunk: string): void;
  
  /**
   * Signal that the stream has ended. Finalizes any incomplete collections.
   */
  end(): void;
  
  /**
   * Get the current partial parse result.
   * This represents the best-effort parse of what has been seen so far.
   */
  readonly partial: ParsedValue | undefined;
  
  /**
   * Whether the parser has completed a top-level value.
   */
  readonly done: boolean;

  /**
   * Get the current nesting path within the JSON structure being parsed.
   * Returns an array of keys/indices representing the current position.
   * e.g. ["config", "tls", "cert"] when parsing inside config.tls.cert
   * Used by the mact parser to compute ToolInputFieldChunk paths.
   */
  readonly currentPath: readonly string[];
}
