/**
 * Finalization Functions
 *
 * Pure functions that convert builders to final AST nodes.
 */

import type {
  ParagraphBuilder,
  HeadingBuilder,
  CodeBlockBuilder,
  HorizontalRuleBuilder,
  HtmlBlockBuilder,
  DefinitionBuilder,
  BlockquoteBuilder,
  BlockquoteBuilderContent,
  BulletListBuilder,
  OrderedListBuilder,
  ListItemBuilder,
  ListBuilderContent,
  TableBuilder,
  TableRowBuilder,
  TableCellBuilder,
  LinkBuilder,
  ImageBuilder,
  EmphasisBuilder,
  StrongBuilder,
  StrikethroughBuilder,
  InlineCodeBuilder,
} from './types'

import type {
  ParagraphNode,
  HeadingNode,
  CodeBlockNode,
  HorizontalRuleNode,
  HtmlBlockNode,
  DefinitionNode,
  BlockquoteNode,
  BlockquoteItemNode,
  BlockquoteItemBreakNode,
  BulletListNode,
  OrderedListNode,
  TaskListNode,
  BulletItemNode,
  OrderedItemNode,
  TaskItemNode,
  ListItemBreakNode,
  BlankLinesNode,
  TableNode,
  TableRowNode,
  TableCellNode,
  LinkNode,
  InlineImageNode,
  EmphasisNode,
  StrongNode,
  StrikethroughNode,
  InlineCodeNode,
  TextNode,
  HardBreakNode,
} from './schema'
import type { SourcePosition, SourcePoint } from './schema'

const ZERO_POINT: SourcePoint = { line: 1, column: 1, offset: 0 }
const ZERO_POSITION: SourcePosition = { start: ZERO_POINT, end: ZERO_POINT }

function pos(builder: { positionStart: SourcePoint | null; positionEnd: SourcePoint | null }): SourcePosition {
  return {
    start: builder.positionStart ?? ZERO_POINT,
    end: builder.positionEnd ?? ZERO_POINT,
  }
}

// =============================================================================
// BLOCK NODE FINALIZERS
// =============================================================================

export function finalizeParagraph(b: ParagraphBuilder): ParagraphNode {
  let content = b.content

  // If there's pending checkbox text (from a [ ] that wasn't a valid checkbox),
  // prepend it to the content
  if (b.pendingCheckboxText) {
    const textNode: TextNode = { type: 'text', text: b.pendingCheckboxText, position: pos(b) }
    content = [textNode, ...content]
  }

  // Check for trailing backslash - convert to hardBreak
  // CommonMark treats trailing \ at end of paragraph as literal, but we want it to be
  // a hardBreak for TipTap roundtrip. See decisions/26-01-06/trailing-backslash-as-hardbreak.md
  if (content.length > 0) {
    const lastNode = content[content.length - 1]
    if (lastNode.type === 'text' && lastNode.text.endsWith('\\')) {
      // Remove trailing backslash from text
      const newText = lastNode.text.slice(0, -1)
      if (newText.length > 0) {
        content = [...content.slice(0, -1), { type: 'text', text: newText, position: pos(b) }]
      } else {
        content = content.slice(0, -1)
      }
      // Add hardBreak at end
      const hardBreak: HardBreakNode = {
        type: 'hardBreak',
        meta: { style: 'backslash', spaces: 0, continuation: '' },
        position: pos(b),
      }
      content = [...content, hardBreak]
    }
  }

  return {
    type: 'paragraph',
    content: content.length > 0 ? content : undefined,
    position: pos(b),
  }
}

export function finalizeHeading(b: HeadingBuilder): HeadingNode {
  if (b.level === null) {
    throw new Error('Heading level not set - missing atxHeadingSequence')
  }
  return {
    type: 'heading',
    attrs: { level: b.level },
    content: b.content.length > 0 ? b.content : undefined,
    meta: {
      openingWhitespace: b.openingWhitespace,
      closingHashes: b.closingHashes,
      trailingWhitespace: b.trailingWhitespace,
    },
    position: pos(b),
  }
}

export function finalizeCodeBlock(b: CodeBlockBuilder): CodeBlockNode {
  const content = b.lines.join('\n')
  return {
    type: 'codeBlock',
    attrs: { language: b.language },
    content: content ? [{ type: 'text', text: content, position: pos(b) }] : undefined,
    meta: {
      fence: b.fence,
      fenceLength: b.fenceLength,
      closingFenceLength: b.closingFenceLength,
      closingFenceIndent: b.closingFenceIndent,
      closed: b.closed,
      infoWhitespace: b.infoWhitespace,
      infoMetaWhitespace: b.infoMetaWhitespace,
      fenceMeta: b.fenceMeta,
    },
    position: pos(b),
  }
}

export function finalizeHorizontalRule(b: HorizontalRuleBuilder): HorizontalRuleNode {
  return {
    type: 'horizontalRule',
    meta: { original: b.original },
    position: pos(b),
  }
}

export function finalizeHtmlBlock(b: HtmlBlockBuilder): HtmlBlockNode {
  return {
    type: 'htmlBlock',
    content: b.content,
    position: pos(b),
  }
}

export function finalizeDefinition(b: DefinitionBuilder): DefinitionNode {
  return {
    type: 'definition',
    label: b.label,
    url: b.url,
    title: b.title,
    meta: {
      hasAngleBrackets: b.hasAngleBrackets,
      titleQuote: b.titleQuote,
      preUrlWhitespace: b.preUrlWhitespace,
      midWhitespace: b.midWhitespace,
    },
    position: pos(b),
  }
}

function isBlockquoteItemBreakNode(item: BlockquoteBuilderContent): item is BlockquoteItemBreakNode {
  return 'type' in item && item.type === 'blockquoteItemBreak'
}

export function finalizeBlockquote(b: BlockquoteBuilder): BlockquoteNode {
  return {
    type: 'blockquote',
    content: b.content.map((item): BlockquoteItemNode | BlockquoteItemBreakNode => {
      if (isBlockquoteItemBreakNode(item)) {
        return item
      }
      return {
        type: 'blockquoteItem' as const,
        content: item.block,
        meta: {
          leadingIndent: item.leadingIndent,
          prefixWhitespace: item.prefixWhitespace,
        },
        position: pos(b),
      }
    }),
    position: pos(b),
  }
}

function isListItemBuilder(item: ListBuilderContent): item is ListItemBuilder {
  return 'builderType' in item && item.builderType === 'listItem'
}

function finalizeBulletItem(b: ListItemBuilder): BulletItemNode {
  const content = b.content.map((item) => ({
    type: 'listItemContentItem' as const,
    content: item.block,
    meta: { indent: item.indent },
    position: pos(b),
  }))

  return {
    type: 'bulletItem',
    content,
    meta: {
      prefixWhitespace: b.prefixWhitespace,
      indent: b.indent,
    },
    position: pos(b),
  }
}

function finalizeOrderedItem(b: ListItemBuilder, number: string): OrderedItemNode {
  const content = b.content.map((item) => ({
    type: 'listItemContentItem' as const,
    content: item.block,
    meta: { indent: item.indent },
    position: pos(b),
  }))

  return {
    type: 'orderedItem',
    content,
    meta: {
      prefixWhitespace: b.prefixWhitespace,
      indent: b.indent,
      number,
    },
    position: pos(b),
  }
}

function finalizeTaskItem(b: ListItemBuilder, number: string | null): TaskItemNode {
  const content = b.content.map((item) => ({
    type: 'listItemContentItem' as const,
    content: item.block,
    meta: { indent: item.indent },
    position: pos(b),
  }))

  const checkbox = b.taskCheckbox
  if (checkbox === null) {
    throw new Error('finalizeTaskItem called on item without checkbox')
  }

  const checked = checkbox === '[x]' || checkbox === '[X]'
  return {
    type: 'taskItem',
    content,
    attrs: { checked },
    meta: {
      prefixWhitespace: b.prefixWhitespace,
      indent: b.indent,
      checkboxMarker: checkbox,
      number,
    },
    position: pos(b),
  }
}

type BulletListResult = BulletListNode | TaskListNode
type OrderedListResult = OrderedListNode | TaskListNode

/** Result of splitting at blank line boundaries */
type BlankLineSplitResult = {
  items: ListItemBuilder[]
  /** BlankLines node to emit BEFORE this group (null for first group) */
  precedingBlankLines: BlankLinesNode | null
}

/**
 * Split list content at blank line boundaries.
 * A ListItemBreakNode indicates blank lines between items - we split there.
 * Returns arrays of consecutive items along with the blank line nodes to emit between them.
 */
function splitAtBlankLineBoundaries(
  content: ListBuilderContent[]
): BlankLineSplitResult[] {
  const results: BlankLineSplitResult[] = []
  let currentItems: ListItemBuilder[] = []
  let pendingBreak: ListItemBreakNode | null = null

  for (const item of content) {
    if (!isListItemBuilder(item)) {
      // ListItemBreakNode - capture it as the separator
      // Finish current group if we have items
      if (currentItems.length > 0) {
        results.push({
          items: currentItems,
          precedingBlankLines: pendingBreak ? {
            type: 'blankLines',
            count: pendingBreak.meta.blankLines.length,
            meta: { lines: [...pendingBreak.meta.blankLines] },
            position: pendingBreak.position,
          } : null,
        })
        currentItems = []
      }
      pendingBreak = item
      continue
    }

    // Regular list item - add to current group
    currentItems.push(item)
  }

  // Don't forget the last group
  if (currentItems.length > 0) {
    results.push({
      items: currentItems,
      precedingBlankLines: pendingBreak && pendingBreak.meta.blankLines.length > 0 ? {
        type: 'blankLines',
        count: pendingBreak.meta.blankLines.length,
        meta: { lines: [...pendingBreak.meta.blankLines] },
        position: pendingBreak.position,
      } : null,
    })
  }

  return results
}

/**
 * Split list items at checkbox boundaries to maintain homogeneous siblings.
 * Returns groups of consecutive items where all items in a group either have
 * checkboxes (task) or don't (regular).
 */
function splitAtCheckboxBoundaries(
  items: ListItemBuilder[]
): Array<{ isTask: boolean; items: ListItemBuilder[] }> {
  const groups: Array<{ isTask: boolean; items: ListItemBuilder[] }> = []
  let currentGroup: { isTask: boolean; items: ListItemBuilder[] } | null = null

  for (const item of items) {
    const isTask = item.taskCheckbox !== null

    if (currentGroup === null || currentGroup.isTask !== isTask) {
      // Start new group
      currentGroup = { isTask, items: [item] }
      groups.push(currentGroup)
    } else {
      // Add to current group
      currentGroup.items.push(item)
    }
  }

  return groups
}

export function finalizeBulletList(b: BulletListBuilder): (BulletListResult | BlankLinesNode)[] {
  // Step 1: Split at blank line boundaries (separate lists)
  const blankLineGroups = splitAtBlankLineBoundaries(b.content)

  // Step 2: For each group, split at checkbox boundaries
  const results: (BulletListResult | BlankLinesNode)[] = []

  for (const blankLineGroup of blankLineGroups) {
    // Emit preceding blank lines if present
    if (blankLineGroup.precedingBlankLines) {
      results.push(blankLineGroup.precedingBlankLines)
    }

    const checkboxGroups = splitAtCheckboxBoundaries(blankLineGroup.items)

    for (const group of checkboxGroups) {
      if (group.isTask) {
        results.push({
          type: 'taskList',
          content: group.items.map((item): TaskItemNode => finalizeTaskItem(item, null)),
          meta: { style: 'bullet', marker: b.marker },
          position: pos(b),
        })
      } else {
        results.push({
          type: 'bulletList',
          content: group.items.map((item): BulletItemNode => finalizeBulletItem(item)),
          meta: { marker: b.marker },
          position: pos(b),
        })
      }
    }
  }

  return results
}

export function finalizeOrderedList(b: OrderedListBuilder): (OrderedListResult | BlankLinesNode)[] {
  // Step 1: Split at blank line boundaries (separate lists)
  const blankLineGroups = splitAtBlankLineBoundaries(b.content)

  // Step 2: For each group, split at checkbox boundaries
  const results: (OrderedListResult | BlankLinesNode)[] = []
  let numberIdx = 0

  for (const blankLineGroup of blankLineGroups) {
    // Emit preceding blank lines if present
    if (blankLineGroup.precedingBlankLines) {
      results.push(blankLineGroup.precedingBlankLines)
    }

    const checkboxGroups = splitAtCheckboxBoundaries(blankLineGroup.items)

    for (const group of checkboxGroups) {
      if (group.isTask) {
        results.push({
          type: 'taskList',
          content: group.items.map((item): TaskItemNode => {
            const number = b.numbers[numberIdx++]
            return finalizeTaskItem(item, number)
          }),
          meta: { style: 'ordered', delimiter: b.delimiter },
          position: pos(b),
        })
      } else {
        results.push({
          type: 'orderedList',
          content: group.items.map((item): OrderedItemNode => {
            const number = b.numbers[numberIdx++]
            return finalizeOrderedItem(item, number)
          }),
          meta: { delimiter: b.delimiter },
          position: pos(b),
        })
      }
    }
  }

  return results
}

// =============================================================================
// TABLE NODE FINALIZERS
// =============================================================================

export function finalizeTableCell(b: TableCellBuilder): TableCellNode {
  const cell: TableCellNode = {
    type: 'tableCell',
    content: [{
      type: 'paragraph',
      content: b.content.length > 0 ? b.content : undefined,
      position: pos(b),
    }],
    meta: {
      leadingWhitespace: b.leadingWhitespace,
      trailingWhitespace: b.trailingWhitespace,
    },
    position: pos(b),
  }

  if (b.alignment) {
    return { ...cell, attrs: { textAlign: b.alignment, colwidth: null } }
  }

  return cell
}

export function finalizeTableRow(b: TableRowBuilder): TableRowNode {
  // Detect leading/trailing pipes from raw row
  const trimmedRow = b.rawRow.trim()
  const hasLeadingPipe = trimmedRow.startsWith('|')
  const hasTrailingPipe = trimmedRow.endsWith('|')

  return {
    type: 'tableRow',
    content: b.cells.map(finalizeTableCell) as [TableCellNode, ...TableCellNode[]],
    meta: {
      hasLeadingPipe,
      hasTrailingPipe,
      linePrefix: b.linePrefix,
    },
    position: pos(b),
  }
}

export function finalizeTable(b: TableBuilder): TableNode {
  return {
    type: 'table',
    content: b.rows.map(finalizeTableRow) as [TableRowNode, ...TableRowNode[]],
    meta: {
      columnWidths: b.columnWidths,
      delimiterRow: b.delimiterRow,
    },
    position: pos(b),
  }
}

// =============================================================================
// INLINE NODE FINALIZERS
// =============================================================================

export function finalizeLink(b: LinkBuilder): LinkNode {
  return {
    type: 'link',
    attrs: { href: b.href, title: b.title },
    content: b.content,
    meta: {
      hasAngleBrackets: b.hasAngleBrackets,
      titleQuote: b.titleQuote,
      preUrlWhitespace: b.preUrlWhitespace,
      midWhitespace: b.midWhitespace,
      postTitleWhitespace: b.postTitleWhitespace,
    },
    position: pos(b),
  }
}

export function finalizeImage(b: ImageBuilder): InlineImageNode {
  return {
    type: 'image',
    attrs: {
      src: b.src,
      alt: b.alt,
      title: b.title,
    },
    meta: {
      hasAngleBrackets: b.hasAngleBrackets,
      titleQuote: b.titleQuote,
      preUrlWhitespace: b.preUrlWhitespace,
      midWhitespace: b.midWhitespace,
      postTitleWhitespace: b.postTitleWhitespace,
    },
    position: pos(b),
  }
}

export function finalizeEmphasis(b: EmphasisBuilder): EmphasisNode {
  return {
    type: 'emphasis',
    meta: { delimiter: b.delimiter },
    content: b.content,
    position: pos(b),
  }
}

export function finalizeStrong(b: StrongBuilder): StrongNode {
  return {
    type: 'strong',
    meta: { delimiter: b.delimiter },
    content: b.content,
    position: pos(b),
  }
}

export function finalizeStrikethrough(b: StrikethroughBuilder): StrikethroughNode {
  return {
    type: 'strikethrough',
    meta: { delimiter: b.delimiter },
    content: b.content,
    position: pos(b),
  }
}

export function finalizeInlineCode(b: InlineCodeBuilder): InlineCodeNode {
  return {
    type: 'inlineCode',
    text: b.content,
    meta: {
      backticks: b.backticks,
      padding: b.padding,
    },
    position: pos(b),
  }
}
