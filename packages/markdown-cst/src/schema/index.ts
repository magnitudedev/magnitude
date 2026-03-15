/**
 * Document Schema
 *
 * Plain TypeScript definitions for DocumentNode - the canonical document representation.
 */

export interface SourcePoint {
  line: number
  column: number
  offset: number
}

export interface SourcePosition {
  start: SourcePoint
  end: SourcePoint
}

// =============================================================================
// SHARED TYPE CONSTANTS
// =============================================================================

/**
 * Valid checkbox markers for task list items.
 * Format: [<value>] where value is whitespace (space, tab, newline) or x/X
 * We use string instead of literal union to support [\t], [\n] for lossless roundtrip.
 */
export type CheckboxMarker = string

/** Valid bullet list markers */
export type BulletMarker = '-' | '*' | '+'

/** Valid ordered list delimiters */
export type OrderedDelimiter = '.' | ')'

/** Valid emphasis delimiters */
export type EmphasisDelimiter = '*' | '_'

/** Valid strong delimiters */
export type StrongDelimiter = '**' | '__'

/** Valid code fence characters */
export type CodeFence = '`' | '~'

/** Valid hard break styles */
export type HardBreakStyle = 'space' | 'backslash'

/** Valid heading levels */
export type HeadingLevel = 1 | 2 | 3 | 4 | 5 | 6

/** Valid text alignments */
export type TextAlign = 'left' | 'center' | 'right'

/** Valid link/image title quote characters */
export type TitleQuote = '"' | "'" | '('

/** Valid strikethrough delimiters */
export type StrikethroughDelimiter = '~' | '~~'

// =============================================================================
// INLINE NODES
// =============================================================================

export interface TextNode {
  type: 'text'
  text: string
  position: SourcePosition
}

export interface HardBreakNode {
  type: 'hardBreak'
  meta: {
    style: HardBreakStyle
    spaces: number
    /** continuation: whitespace that appears at the start of the next line (e.g., "     " in "foo  \n     bar") */
    continuation: string
  }
  position: SourcePosition
}

export interface SoftBreakNode {
  type: 'softBreak'
  /** continuation: whitespace that appears at the start of the next line (e.g., "  " or "\t") */
  meta: {
    continuation: string
  }
  position: SourcePosition
}

export interface InlineImageNode {
  type: 'image'
  attrs: {
    src: string
    alt: string | null
    title: string | null
  }
  meta: {
    /** Whether the URL was wrapped in angle brackets (<url>) */
    hasAngleBrackets: boolean
    /** Quote character used for title, if any */
    titleQuote: TitleQuote | null
    /** Whitespace/newlines before the URL (after opening paren) */
    preUrlWhitespace: string
    /** Whitespace/newlines between URL and title (or before closing paren if no title) */
    midWhitespace: string
    /** Whitespace after title (before closing paren) */
    postTitleWhitespace: string
  }
  position: SourcePosition
}

// DISABLED: Math support temporarily disabled
// export interface InlineMathNode {
//   type: 'inlineMath'
//   attrs: {
//     latex: string
//   }
//   position: SourcePosition
// }

export interface InlineCodeNode {
  type: 'inlineCode'
  text: string
  meta: {
    backticks: number
    /** Padding character if padded (space or newline), empty string if not padded */
    padding: string
  }
  position: SourcePosition
}

export interface EmphasisNode {
  type: 'emphasis'
  meta: {
    delimiter: EmphasisDelimiter
  }
  content: readonly InlineNode[]
  position: SourcePosition
}

export interface StrongNode {
  type: 'strong'
  meta: {
    delimiter: StrongDelimiter
  }
  content: readonly InlineNode[]
  position: SourcePosition
}

export interface StrikethroughNode {
  type: 'strikethrough'
  meta: {
    /** delimiter: "~" (single) or "~~" (double) */
    delimiter: StrikethroughDelimiter
  }
  content: readonly InlineNode[]
  position: SourcePosition
}

export interface LinkNode {
  type: 'link'
  attrs: {
    href: string
    title: string | null
  }
  content: readonly InlineNode[]
  meta: {
    /** Whether the URL was wrapped in angle brackets (<url>) */
    hasAngleBrackets: boolean
    /** Quote character used for title, if any */
    titleQuote: TitleQuote | null
    /** Whitespace/newlines before the URL (after opening paren) */
    preUrlWhitespace: string
    /** Whitespace/newlines between URL and title (or before closing paren if no title) */
    midWhitespace: string
    /** Whitespace after title (before closing paren) */
    postTitleWhitespace: string
  }
  position: SourcePosition
}

export type InlineNode =
  | TextNode
  | HardBreakNode
  | SoftBreakNode
  | InlineImageNode
  // DISABLED: Math support temporarily disabled
  // | InlineMathNode
  | InlineCodeNode
  | EmphasisNode
  | StrongNode
  | StrikethroughNode
  | LinkNode

// =============================================================================
// LEAF BLOCK NODES (non-recursive)
// =============================================================================

export interface ParagraphNode {
  type: 'paragraph'
  content?: readonly InlineNode[]
  position: SourcePosition
}

export interface HeadingNode {
  type: 'heading'
  attrs: {
    level: HeadingLevel
  }
  content?: readonly InlineNode[]
  meta: {
    /** Whitespace between opening # and content */
    openingWhitespace: string
    /** Closing hashes including preceding whitespace (e.g., " ##") */
    closingHashes: string
    /** Trailing whitespace after content/closing hashes */
    trailingWhitespace: string
  }
  position: SourcePosition
}

export interface CodeBlockNode {
  type: 'codeBlock'
  attrs: {
    language: string | null
  }
  content?: readonly TextNode[]
  meta: {
    fence: CodeFence
    fenceLength: number
    /** Closing fence length (may differ from opening) */
    closingFenceLength: number
    /** Leading indent of closing fence (whitespace before ``` on closing line) */
    closingFenceIndent: string
    closed: boolean
    /** Whitespace between fence sequence and info string (e.g., " " in "``` foo") */
    infoWhitespace: string
    /** Whitespace between info and meta (e.g., " " in "~~~ aa ``` ~~~") */
    infoMetaWhitespace: string
    /** Fence meta string - content after info/language (e.g., "``` ~~~" in "~~~ aa ``` ~~~") */
    fenceMeta: string
  }
  position: SourcePosition
}

export interface HorizontalRuleNode {
  type: 'horizontalRule'
  meta: {
    original: string
  }
  position: SourcePosition
}

export interface ImageBlockNode {
  type: 'image'
  attrs: {
    src: string
    alt: string | null
    title: string | null
  }
  position: SourcePosition
}

// DISABLED: Math support temporarily disabled
// export interface BlockMathNode {
//   type: 'blockMath'
//   attrs: {
//     latex: string
//   }
//   position: SourcePosition
// }

/** Raw HTML block - preserved as-is for lossless round-trip */
export interface HtmlBlockNode {
  type: 'htmlBlock'
  content: string
  position: SourcePosition
}

/** Link reference definition - preserved for lossless round-trip */
export interface DefinitionNode {
  type: 'definition'
  /** The label (identifier) */
  label: string
  /** The destination URL */
  url: string
  /** Optional title */
  title: string | null
  meta: {
    /** Whether URL was wrapped in angle brackets */
    hasAngleBrackets: boolean
    /** Quote character for title */
    titleQuote: TitleQuote | null
    /** Whitespace before URL */
    preUrlWhitespace: string
    /** Whitespace between URL and title */
    midWhitespace: string
  }
  position: SourcePosition
}

// =============================================================================
// TABLE
// =============================================================================

export interface TableCellNode {
  type: 'tableCell'
  attrs?: {
    colspan?: number
    rowspan?: number
    colwidth?: readonly number[] | null
    textAlign?: TextAlign | null
  }
  content: [ParagraphNode]
  meta: {
    leadingWhitespace: string
    trailingWhitespace: string
  }
  position: SourcePosition
}

export interface TableRowNode {
  type: 'tableRow'
  content: readonly [TableCellNode, ...TableCellNode[]]
  meta: {
    hasLeadingPipe: boolean
    hasTrailingPipe: boolean
    /** Line prefix for rows inside containers (blockquote marker, list indent) */
    linePrefix: string
  }
  position: SourcePosition
}

export interface TableNode {
  type: 'table'
  content: readonly [TableRowNode, ...TableRowNode[]]
  meta: {
    columnWidths: readonly number[]
    /** Raw delimiter row for lossless serialization */
    delimiterRow: string
  }
  position: SourcePosition
}

// =============================================================================
// BLANK LINES NODE
// =============================================================================

export interface BlankLinesNode {
  type: 'blankLines'
  count: number
  meta: {
    /**
     * For lossless round-trip: the content of each blank line (whitespace).
     * lines[i] is the content of blank line i (may be empty or whitespace-only).
     */
    lines: readonly string[]
  }
  position: SourcePosition
}

// =============================================================================
// RECURSIVE BLOCK NODES (lists, blockquotes)
// =============================================================================

/** Content allowed in a list item (paragraph, nested lists, blank lines) */
export type ListItemContentNode =
  | ParagraphNode
  | BulletListNode
  | OrderedListNode
  | TaskListNode
  | BlankLinesNode

/** Wrapper for each content block inside a list item */
export interface ListItemContentItemNode {
  type: 'listItemContentItem'
  content: ListItemContentNode
  meta: {
    /** indent: full indent before this block (continuation indent from line start) */
    indent: string
  }
  position: SourcePosition
}

/** Bullet list item */
export interface BulletItemNode {
  type: 'bulletItem'
  content: readonly ListItemContentItemNode[]
  meta: {
    /** prefixWhitespace: whitespace between marker and content (e.g., "  " in "-  foo") */
    prefixWhitespace: string
    /** indent: full indent before the list item marker (structural + extra combined) */
    indent: string
  }
  position: SourcePosition
}

/** Ordered list item */
export interface OrderedItemNode {
  type: 'orderedItem'
  content: readonly ListItemContentItemNode[]
  meta: {
    /** prefixWhitespace: whitespace between marker and content (e.g., "  " in "1.  foo") */
    prefixWhitespace: string
    /** indent: full indent before the list item marker (structural + extra combined) */
    indent: string
    /** number: original number string for lossless roundtrip (e.g., '1', '02', '10') */
    number: string
  }
  position: SourcePosition
}

/** Task list item with checkbox */
export interface TaskItemNode {
  type: 'taskItem'
  content: readonly ListItemContentItemNode[]
  attrs: { checked: boolean }
  meta: {
    /** prefixWhitespace: whitespace between checkbox and content */
    prefixWhitespace: string
    /** indent: full indent before the list item marker (structural + extra combined) */
    indent: string
    /** Original checkbox marker for lossless roundtrip */
    checkboxMarker: CheckboxMarker
    /** number: original number string if from ordered-style, null if from bullet-style */
    number: string | null
  }
  position: SourcePosition
}

/** Break between list items - captures blank lines and continuation prefix */
export interface ListItemBreakNode {
  type: 'listItemBreak'
  meta: {
    /** Content of each blank line between items (empty array for tight lists) */
    blankLines: readonly string[]
    /** Prefix before the next item's marker line (e.g., "> " for blockquote) */
    continuation: string
  }
  position: SourcePosition
}

export interface BulletListNode {
  type: 'bulletList'
  content: readonly (BulletItemNode | ListItemBreakNode)[]
  meta: { marker: BulletMarker }
  position: SourcePosition
}

export interface OrderedListNode {
  type: 'orderedList'
  content: readonly (OrderedItemNode | ListItemBreakNode)[]
  meta: { delimiter: OrderedDelimiter }
  position: SourcePosition
}

/** Task list meta - discriminated union for bullet vs ordered style */
export type TaskListMeta =
  | { style: 'bullet'; marker: BulletMarker }
  | { style: 'ordered'; delimiter: OrderedDelimiter }

export interface TaskListNode {
  type: 'taskList'
  content: readonly (TaskItemNode | ListItemBreakNode)[]
  meta: TaskListMeta
  position: SourcePosition
}

export type BlockquoteContentNode =
  | ParagraphNode
  | HeadingNode
  | BulletListNode
  | OrderedListNode
  | TaskListNode
  | BlockquoteNode
  | BlankLinesNode

export interface BlockquoteItemNode {
  type: 'blockquoteItem'
  content: BlockquoteContentNode
  meta: {
    /** leadingIndent: whitespace before the > marker (e.g., "   " in "   > foo") */
    leadingIndent: string
    /** prefixWhitespace: whitespace after > marker (e.g., " " in "> foo", or "" in ">foo") */
    prefixWhitespace: string
  }
  position: SourcePosition
}

/** Break between blockquote items - captures blank lines and full prefix chain for next item */
export interface BlockquoteItemBreakNode {
  type: 'blockquoteItemBreak'
  meta: {
    /** Full prefix strings for blank lines (e.g., [">>", ">>"]) */
    blankLines: readonly string[]
    /** Full prefix chain for next content's first line (e.g., "  >  > ") */
    continuation: string
  }
  position: SourcePosition
}

export interface BlockquoteNode {
  type: 'blockquote'
  content: readonly (BlockquoteItemNode | BlockquoteItemBreakNode)[]
  position: SourcePosition
}

// =============================================================================
// ROOT/DOCUMENT TYPES
// =============================================================================

export type RootBlockNode =
  | ParagraphNode
  | HeadingNode
  | BulletListNode
  | OrderedListNode
  | TaskListNode
  | CodeBlockNode
  | BlockquoteNode
  | HorizontalRuleNode
  | ImageBlockNode
  | TableNode
  // DISABLED: Math support temporarily disabled
  // | BlockMathNode
  | HtmlBlockNode
  | DefinitionNode

export type DocumentContentNode = RootBlockNode | BlankLinesNode

export interface DocumentItemNode {
  type: 'documentItem'
  content: DocumentContentNode
  meta: {
    /** leadingIndent: whitespace before this block (e.g., "   " in "   # Heading") */
    leadingIndent: string
  }
  position: SourcePosition
}

export interface DocumentNode {
  type: 'doc'
  content: readonly DocumentItemNode[]
  meta: {
    /** Whether the source ended with a trailing newline */
    trailingNewline: boolean
  }
  position: SourcePosition
  source: string
}

// =============================================================================
// UTILITIES
// =============================================================================

export function emptyDocument(): DocumentNode {
  const zeroPoint: SourcePoint = { line: 1, column: 1, offset: 0 }
  const zeroPos: SourcePosition = { start: zeroPoint, end: zeroPoint }
  return {
    type: 'doc',
    content: [],
    meta: { trailingNewline: false },
    position: zeroPos,
    source: '',
  }
}

export function isEmptyDocument(doc: DocumentNode): boolean {
  return doc.content.length === 0
}

// =============================================================================
// DECORATIVE NODES
// =============================================================================

/**
 * Decorative nodes exist purely for lossless markdown roundtrip.
 * They have no representation in tiptap - removing them has zero impact
 * on the converted tiptap document.
 */
export const DECORATIVE_NODE_TYPES = new Set([
  'listItemBreak',
  'blockquoteItemBreak',
] as const)

export type DecorativeNodeType = typeof DECORATIVE_NODE_TYPES extends Set<infer T> ? T : never

export function isDecorativeNode(node: { type: string }): boolean {
  return DECORATIVE_NODE_TYPES.has(node.type as DecorativeNodeType)
}