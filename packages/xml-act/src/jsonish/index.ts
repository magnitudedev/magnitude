
/**
 * JSONish streaming parser module
 * 
 * Provides incremental JSON parsing for LLM-generated tool call parameters.
 * Ported from BAML's jsonish parser with simplifications for our use case.
 */

export type {
  CompletionState,
  ParsedValue,
  ParsedString,
  ParsedNumber,
  ParsedBoolean,
  ParsedNull,
  ParsedObject,
  ParsedArray,
  StreamingJsonParser,
} from './types';

export { createStreamingJsonParser } from './parser'

export type {
  CoercionFlag,
  CoercedResult,
} from './coercer'

export { coerceToStreamingPartial, tryCastToStreamingPartial } from './coercer'

export { createParameterAccumulator } from './accumulator';
