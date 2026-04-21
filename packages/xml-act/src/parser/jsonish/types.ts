// Re-export from the root jsonish types — same types, just accessible from parser/jsonish/
export type {
  CompletionState,
  ParsedValue,
  ParsedString,
  ParsedNumber,
  ParsedBoolean,
  ParsedNull,
  ParsedObject,
  ParsedArray,
  JsonCollection,
  ObjectCollection,
  ArrayCollection,
  QuotedStringCollection,
  UnquotedStringCollection,
  CloseStringResult,
  Pos,
  StreamingJsonParser,
} from '../../jsonish/types'
