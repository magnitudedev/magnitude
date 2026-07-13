/**
 * Parser Types
 *
 * Type-safe builder-based architecture for converting micromark events to AST.
 *
 * Key concepts:
 * - Builders accumulate data during parsing and are finalized into nodes
 * - Each builder type is discriminated via builderType field
 * - Context tracks builder stack with proper typing
 * - Handlers work with strongly-typed builders
 */

import type { Token, TokenForType, TokenType } from './tokenizer'
import type {
  InlineNode,
  TextNode,
  SourcePoint,
  SourcePosition,
  BlankLinesNode,
  BulletItemNode,
  OrderedItemNode,
  TaskItemNode,
  ListItemBreakNode,
  ListItemContentNode,
  ListItemContentItemNode,
  BlockquoteContentNode,
  BlockquoteItemBreakNode,
  DocumentContentNode,
  TitleQuote,
  HeadingLevel,
  BulletMarker,
  OrderedDelimiter,
  CodeFence,
  EmphasisDelimiter,
  StrongDelimiter,
  StrikethroughDelimiter,
  HardBreakStyle,
  CheckboxMarker,
} from './schema'

// =============================================================================
// BUILDER TYPES
// =============================================================================

/**
 * Base builder interface - all builders have a type discriminant
 */
interface BaseBuilder<T extends string> {
  readonly builderType: T
  positionStart: SourcePoint | null
  positionEnd: SourcePoint | null
}

/** Mutable text node for building (schema TextNode is readonly) */
export interface MutableTextNode {
  type: 'text'
  text: string
  position: SourcePosition
}

/**
 * Inline content accumulator - shared by builders that collect inline nodes
 */
interface InlineContentTrait {
  content: InlineNode[]
  /** Currently open text node being appended to */
  currentText: MutableTextNode | null
  /** Pending soft break - waiting to accumulate continuation prefix */
  pendingSoftBreak: { continuation: string } | null
  /** Pending hard break - waiting to accumulate continuation prefix */
  pendingHardBreak: { style: 'space' | 'backslash'; spaces: number; continuation: string } | null
}

// -----------------------------------------------------------------------------
// Block Builders
// -----------------------------------------------------------------------------

export interface ParagraphBuilder extends BaseBuilder<'paragraph'>, InlineContentTrait {
  /** Pending checkbox text to prepend (when [ ] appears but isn't a valid checkbox) */
  pendingCheckboxText?: string
}

export interface HeadingBuilder extends BaseBuilder<'heading'>, InlineContentTrait {
  /** null until opening sequence is processed */
  level: HeadingLevel | null
  openingWhitespace: string
  /** Closing hashes including preceding whitespace (e.g., " ##") */
  closingHashes: string
  /** Trailing whitespace after closing hashes */
  trailingWhitespace: string
  /** Buffer for whitespace that might be before closing hashes or trailing */
  pendingWhitespace: string
  /** For setext headings - skip line ending after heading text */
  slurpLineEnding: boolean
}

export interface SetextHeadingBuilder extends BaseBuilder<'setextHeading'> {
  /** Leading whitespace before heading text */
  leadingIndent: string
  /** Accumulated text content */
  text: string
  /** Whitespace before the underline */
  underlineIndent: string
  /** The full underline line (sequence + trailing whitespace) */
  underline: string
}

export interface CodeBlockBuilder extends BaseBuilder<'codeBlock'> {
  language: string | null
  fence: CodeFence
  fenceLength: number
  closingFenceLength: number
  closingFenceIndent: string
  fenceMeta: string
  infoWhitespace: string
  infoMetaWhitespace: string
  /** Buffered content lines */
  lines: string[]
  /** Whether we're inside the code (past opening fence) */
  insideCode: boolean
  /** Whether we're currently processing the closing fence */
  inClosingFence: boolean
  /** Whether we've seen an exit event for fencedCode (not just codeFenced) */
  closed: boolean
}

export interface HorizontalRuleBuilder extends BaseBuilder<'horizontalRule'> {
  original: string
}

export interface HtmlBlockBuilder extends BaseBuilder<'htmlBlock'> {
  content: string
}

export interface DefinitionBuilder extends BaseBuilder<'definition'> {
  label: string
  url: string
  title: string | null
  hasAngleBrackets: boolean
  titleQuote: TitleQuote | null
  preUrlWhitespace: string
  midWhitespace: string
}

/** Pending break between blockquote items - accumulates until next content */
export interface PendingBlockquoteBreak {
  /** Full prefix strings for blank lines */
  blankLines: string[]
  /** Full prefix chain being accumulated for next content's first line */
  continuation: string
}

/** Content of blockquote builder - items and breaks interleaved */
export type BlockquoteBuilderContent = {
  block: BlockquoteContentNode
  leadingIndent: string
  prefixWhitespace: string
} | BlockquoteItemBreakNode

export interface BlockquoteBuilder extends BaseBuilder<'blockquote'> {
  /** Content blocks collected (wrapped in BlockquoteItemNode at finalization) */
  content: BlockquoteBuilderContent[]
  /** Track current line's metadata */
  currentLeadingIndent: string
  currentPrefixWhitespace: string
  /** Pending break info to become BlockquoteItemBreakNode when next content starts */
  pendingBreak: PendingBlockquoteBreak | null
  /** Prefix chain being accumulated for current line (reset on lineEnding) */
  currentPrefixChain: string
  /** Whether we've seen a prefix on the current line (to distinguish leading vs trailing linePrefix) */
  seenPrefixOnLine: boolean
  /** Whether content was added after the last prefix (to know if trailing prefix needs to become blank line) */
  contentAddedAfterPrefix: boolean
}

/** Pending break between list items - accumulates until next item or list exit */
export interface PendingListItemBreak {
  /** Content of each blank line between items */
  blankLines: string[]
  /** Prefix before the next item's marker line (accumulated from blockQuotePrefix, linePrefix, etc.) */
  continuation: string
}

/** Content of list builder - items and breaks interleaved */
export type ListBuilderContent = ListItemBuilder | ListItemBreakNode

export interface BulletListBuilder extends BaseBuilder<'bulletList'> {
  marker: BulletMarker
  /** Item builders and break nodes - finalized when list exits */
  content: ListBuilderContent[]
  /** Pending break info to become ListItemBreakNode when next item starts */
  pendingBreak: PendingListItemBreak | null
}

export interface OrderedListBuilder extends BaseBuilder<'orderedList'> {
  start: number
  delimiter: OrderedDelimiter
  numbers: string[]
  /** Item builders and break nodes - finalized when list exits */
  content: ListBuilderContent[]
  /** Whether we're expecting the first item value (to set start) */
  expectingFirstValue: boolean
  /** Pending break info to become ListItemBreakNode when next item starts */
  pendingBreak: PendingListItemBreak | null
}

export interface ListItemBuilder extends BaseBuilder<'listItem'> {
  prefixWhitespace: string
  /** Full indent before the list item marker */
  indent: string
  content: Array<{
    block: ListItemContentNode
    /** Full indent before this content block (from line start) */
    indent: string
  }>
  /** Pending indent from listItemIndent - to be applied to next block */
  pendingIndent: string
  /** Pending blockquote prefix - tracked separately from indent to avoid double-counting */
  pendingBlockquotePrefix: string
  /** Pending blank lines count */
  pendingBlankLines: number
  /** Whether we've seen the first lineEndingBlank after marker (marker line ending vs actual blank line) */
  seenMarkerLineEnding: boolean
  /** If this item has a task checkbox, stores the marker; null if regular list item */
  taskCheckbox: CheckboxMarker | null
  /** Temporary: whether a pending checkbox is valid (at start of content) */
  _pendingCheckboxValid?: boolean
  /** Temporary: the pending checkbox marker before we decide to use it */
  _pendingCheckboxMarker?: string
}

// -----------------------------------------------------------------------------
// Table Builders
// -----------------------------------------------------------------------------

export interface TableBuilder extends BaseBuilder<'table'> {
  /** Accumulated row builders */
  rows: TableRowBuilder[]
  /** Column alignments from delimiter row */
  columnAlignments: Array<'left' | 'center' | 'right' | null>
  /** Column widths (filler length) from delimiter row */
  columnWidths: number[]
  /** Raw delimiter row string for lossless serialization */
  delimiterRow: string
  /** Pending line prefix (blockquote marker, list indent) for next row */
  pendingLinePrefix: string
}

export interface TableRowBuilder extends BaseBuilder<'tableRow'> {
  /** Accumulated cell builders */
  cells: TableCellBuilder[]
  /** Raw row string for detecting leading/trailing pipes */
  rawRow: string
  /** Line prefix (container indent) */
  linePrefix: string
}

export interface TableCellBuilder extends BaseBuilder<'tableCell'>, InlineContentTrait {
  /** Alignment for this cell (from delimiter row) */
  alignment: 'left' | 'center' | 'right' | null
  /** Whitespace before content */
  leadingWhitespace: string
  /** Whitespace after content */
  trailingWhitespace: string
  /** Whether we've seen content (to distinguish leading vs trailing whitespace) */
  sawContent: boolean
}

export interface TableDelimiterBuilder extends BaseBuilder<'tableDelimiter'> {
  /** Whether we've seen left colon */
  hasLeftColon: boolean
  /** Whether we've seen right colon */
  hasRightColon: boolean
  /** Filler width (number of dashes) */
  fillerWidth: number
}

// -----------------------------------------------------------------------------
// Inline Builders
// -----------------------------------------------------------------------------

export interface LinkBuilder extends BaseBuilder<'link'>, InlineContentTrait {
  href: string
  title: string | null
  hasAngleBrackets: boolean
  titleQuote: TitleQuote | null
  preUrlWhitespace: string
  midWhitespace: string
  postTitleWhitespace: string
  /** Resource parsing state */
  seenUrl: boolean
  seenTitle: boolean
}

export interface ImageBuilder extends BaseBuilder<'image'> {
  src: string
  alt: string | null
  title: string | null
  hasAngleBrackets: boolean
  titleQuote: TitleQuote | null
  preUrlWhitespace: string
  midWhitespace: string
  postTitleWhitespace: string
  /** Resource parsing state */
  seenUrl: boolean
  seenTitle: boolean
}

export interface EmphasisBuilder extends BaseBuilder<'emphasis'>, InlineContentTrait {
  delimiter: EmphasisDelimiter
}

export interface StrongBuilder extends BaseBuilder<'strong'>, InlineContentTrait {
  delimiter: StrongDelimiter
}

export interface StrikethroughBuilder extends BaseBuilder<'strikethrough'>, InlineContentTrait {
  delimiter: StrikethroughDelimiter
}

export interface InlineCodeBuilder extends BaseBuilder<'inlineCode'> {
  backticks: number
  content: string
  padding: string
}

export interface HardBreakBuilder extends BaseBuilder<'hardBreak'> {
  style: HardBreakStyle
  spaces: number
  continuation: string
}

export interface SoftBreakBuilder extends BaseBuilder<'softBreak'> {
  continuation: string
}

// -----------------------------------------------------------------------------
// Special Builders
// -----------------------------------------------------------------------------

/** Fragment for buffering text content */
export interface FragmentBuilder extends BaseBuilder<'fragment'> {
  text: string
}

/** Raw block builder - captures source text for unsupported blocks in restricted contexts */
export interface RawBlockBuilder extends BaseBuilder<'rawBlock'> {
  /** Start offset in source for slicing raw text */
  startOffset: number
  /** The original token type being captured */
  originalType: string
}

/** Temporary container for document-level content */
export interface ContainerBuilder extends BaseBuilder<'container'> {
  content: Array<{
    block: DocumentContentNode
    leadingIndent: string
  }>
}

// -----------------------------------------------------------------------------
// Union Type
// -----------------------------------------------------------------------------

export type Builder =
  // Blocks
  | ParagraphBuilder
  | HeadingBuilder
  | SetextHeadingBuilder
  | CodeBlockBuilder
  | HorizontalRuleBuilder
  | HtmlBlockBuilder
  | DefinitionBuilder
  | BlockquoteBuilder
  | BulletListBuilder
  | OrderedListBuilder
  | ListItemBuilder
  // Tables
  | TableBuilder
  | TableRowBuilder
  | TableCellBuilder
  | TableDelimiterBuilder
  // Inline
  | LinkBuilder
  | ImageBuilder
  | EmphasisBuilder
  | StrongBuilder
  | StrikethroughBuilder
  | InlineCodeBuilder
  | HardBreakBuilder
  | SoftBreakBuilder
  // Special
  | FragmentBuilder
  | RawBlockBuilder
  | ContainerBuilder

export type BuilderType = Builder['builderType']

// =============================================================================
// CONTEXT
// =============================================================================

export interface CompileContext {
  /** Stack of builders being constructed */
  readonly stack: Builder[]

  /** Stack of tokens for matching enter/exit */
  readonly tokenStack: Array<{ token: Token; builderType: BuilderType }>

  /** Source string for slicing */
  readonly source: string

  /** Current token being dispatched */
  currentToken: Token | null

  // -------------------------------------------------------------------------
  // Builder Operations
  // -------------------------------------------------------------------------

  /** Push a new builder onto the stack */
  push<T extends Builder>(builder: T): T

  /** Pop the top builder, verify it matches expected type */
  pop<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }>

  /** Get the current (top) builder with type narrowing */
  current<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }> | null

  /** Get current builder, throw if doesn't match type */
  require<T extends BuilderType>(expectedType: T): Extract<Builder, { builderType: T }>

  /** Find a builder in the stack by type (searches from top) */
  find<T extends BuilderType>(type: T): Extract<Builder, { builderType: T }> | null

  // -------------------------------------------------------------------------
  // Token Operations
  // -------------------------------------------------------------------------

  /** Record that we entered a token */
  enterToken(token: Token, builderType: BuilderType): void

  /** Record that we exited a token, verify match */
  exitToken(token: Token): void

  /** Get string content for a token */
  slice(token: Token): string

  // -------------------------------------------------------------------------
  // Text Buffering
  // -------------------------------------------------------------------------

  /** Start a text buffer (push FragmentBuilder) */
  buffer(): void

  /** End text buffer and return accumulated text */
  resume(): string

  // -------------------------------------------------------------------------
  // Inline Content Helpers
  // -------------------------------------------------------------------------

  /** Append text to the current inline content builder */
  appendText(text: string): void

  /** Flush current text node and prepare for non-text content */
  flushText(): void

  /** Add an inline node to current inline content builder */
  addInline(node: InlineNode): void

  // -------------------------------------------------------------------------
  // Break Handling (soft and hard)
  // -------------------------------------------------------------------------

  /** Start a pending soft break (for line ending within inline content) */
  startSoftBreak(): void

  /** Start a pending hard break (for trailing spaces or backslash line ending) */
  startHardBreak(style: 'space' | 'backslash', spaces: number): void

  /** Append text to pending break continuation (soft or hard, returns true if there was a pending break) */
  appendContinuation(text: string): boolean

  /** Check if there's a pending soft break */
  hasPendingSoftBreak(): boolean

  /** Check if there's a pending hard break */
  hasPendingHardBreak(): boolean
}

// =============================================================================
// HANDLERS
// =============================================================================

/**
 * Handler function type - receives context and token
 */
export type Handler<T extends Token = Token> = (ctx: CompileContext, token: T) => void

/**
 * Exhaustive handler map - must have a handler for every token type.
 * Use no-op `() => {}` for tokens that should be explicitly ignored.
 */
export type Handlers = {
  [K in TokenType]: (ctx: CompileContext, token: TokenForType<K>) => void
}

/**
 * Handler configuration - maps token types to handlers
 */
export interface HandlerConfig {
  enter: Handlers
  exit: Handlers
}

// =============================================================================
// COMPILE DATA
// =============================================================================

/**
 * Data shared across compilation
 */
export interface CompileData {
  /** Link/image reference definitions for resolving references */
  definitions: Map<string, { url: string; title: string | null }>
}
