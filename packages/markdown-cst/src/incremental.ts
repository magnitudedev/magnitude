import type {
  BlankLinesNode,
  BlockquoteContentNode,
  BlockquoteItemBreakNode,
  BlockquoteItemNode,
  BlockquoteNode,
  BulletItemNode,
  BulletListNode,
  CodeBlockNode,
  DefinitionNode,
  DocumentItemNode,
  DocumentNode,
  HeadingNode,
  HorizontalRuleNode,
  HtmlBlockNode,
  ImageBlockNode,
  InlineCodeNode,
  InlineImageNode,
  InlineNode,
  LinkNode,
  ListItemBreakNode,
  ListItemContentItemNode,
  ListItemContentNode,
  OrderedItemNode,
  OrderedListNode,
  ParagraphNode,
  RootBlockNode,
  SoftBreakNode,
  HardBreakNode,
  SourcePoint,
  SourcePosition,
  StrongNode,
  StrikethroughNode,
  EmphasisNode,
  TableCellNode,
  TableNode,
  TableRowNode,
  TaskItemNode,
  TaskListNode,
  TextNode,
} from './schema'

export function findDivergence(oldSource: string, newSource: string): number | null {
  const limit = Math.min(oldSource.length, newSource.length)

  for (let index = 0; index < limit; index++) {
    if (oldSource[index] !== newSource[index]) {
      return index
    }
  }

  return oldSource.length === newSource.length ? null : limit
}

export function findStablePrefixCount(
  previous: DocumentNode,
  divergeAt: number
): { stableCount: number; cutPoint: number } {
  // We can only safely reuse items up to a clean block boundary.
  // In the AST, blankLines nodes are explicit block separators.
  // We cut after the last blankLines node that ends before divergeAt,
  // ensuring the tail reparse starts at a clean "start of new block" state.
  //
  // Items are stable if they end before divergeAt (their source text is
  // identical in old and new). But we only set the cutPoint after a
  // blankLines item, because that's the only position where the tail
  // parser produces structurally identical results to a full parse.

  let stableCount = 0
  let cutPoint = 0

  for (let i = 0; i < previous.content.length; i++) {
    const item = previous.content[i]

    if (item.content.position.end.offset > divergeAt) {
      break
    }

    stableCount = i + 1

    // Only update cutPoint when we're at a blankLines node —
    // this is a structural block boundary in the AST.
    if (item.content.type === 'blankLines') {
      cutPoint = item.content.position.end.offset
    }
  }

  // If we consumed items but never hit a blankLines boundary,
  // we can't safely cut — reparse everything.
  if (cutPoint === 0) {
    return { stableCount: 0, cutPoint: 0 }
  }

  // Trim stableCount back to only include items up through
  // the last blankLines boundary we found.
  let trimmedCount = 0
  for (let i = 0; i < stableCount; i++) {
    trimmedCount = i + 1
    if (previous.content[i].content.position.end.offset >= cutPoint) {
      break
    }
  }

  return { stableCount: trimmedCount, cutPoint }
}

export function countNewlines(source: string, start: number, end: number): number {
  let count = 0

  for (let index = start; index < end; index++) {
    if (source[index] === '\n') {
      count++
    }
  }

  return count
}

function rebasePoint(point: SourcePoint, offsetDelta: number, lineDelta: number): SourcePoint {
  return {
    offset: point.offset + offsetDelta,
    line: point.line + lineDelta,
    column: point.column,
  }
}

function rebasePosition(
  position: SourcePosition,
  offsetDelta: number,
  lineDelta: number
): SourcePosition {
  return {
    start: rebasePoint(position.start, offsetDelta, lineDelta),
    end: rebasePoint(position.end, offsetDelta, lineDelta),
  }
}

function rebaseTextNode(node: TextNode, offsetDelta: number, lineDelta: number): TextNode {
  return {
    ...node,
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseInlineNode(node: InlineNode, offsetDelta: number, lineDelta: number): InlineNode {
  switch (node.type) {
    case 'text':
      return rebaseTextNode(node, offsetDelta, lineDelta)
    case 'hardBreak':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies HardBreakNode
    case 'softBreak':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies SoftBreakNode
    case 'image':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies InlineImageNode
    case 'inlineCode':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies InlineCodeNode
    case 'emphasis':
      return {
        ...node,
        content: node.content.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies EmphasisNode
    case 'strong':
      return {
        ...node,
        content: node.content.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies StrongNode
    case 'strikethrough':
      return {
        ...node,
        content: node.content.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies StrikethroughNode
    case 'link':
      return {
        ...node,
        content: node.content.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies LinkNode
  }
}

function rebaseParagraphNode(
  node: ParagraphNode,
  offsetDelta: number,
  lineDelta: number
): ParagraphNode {
  return {
    ...node,
    content: node.content?.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseHeadingNode(node: HeadingNode, offsetDelta: number, lineDelta: number): HeadingNode {
  return {
    ...node,
    content: node.content?.map(child => rebaseInlineNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseCodeBlockNode(
  node: CodeBlockNode,
  offsetDelta: number,
  lineDelta: number
): CodeBlockNode {
  return {
    ...node,
    content: node.content?.map(child => rebaseTextNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseTableCellNode(
  node: TableCellNode,
  offsetDelta: number,
  lineDelta: number
): TableCellNode {
  return {
    ...node,
    content: [rebaseParagraphNode(node.content[0], offsetDelta, lineDelta)],
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseTableRowNode(
  node: TableRowNode,
  offsetDelta: number,
  lineDelta: number
): TableRowNode {
  const [first, ...rest] = node.content

  return {
    ...node,
    content: [
      rebaseTableCellNode(first, offsetDelta, lineDelta),
      ...rest.map(cell => rebaseTableCellNode(cell, offsetDelta, lineDelta)),
    ],
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseTableNode(node: TableNode, offsetDelta: number, lineDelta: number): TableNode {
  const [first, ...rest] = node.content

  return {
    ...node,
    content: [
      rebaseTableRowNode(first, offsetDelta, lineDelta),
      ...rest.map(row => rebaseTableRowNode(row, offsetDelta, lineDelta)),
    ],
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseListItemContentNode(
  node: ListItemContentNode,
  offsetDelta: number,
  lineDelta: number
): ListItemContentNode {
  switch (node.type) {
    case 'paragraph':
      return rebaseParagraphNode(node, offsetDelta, lineDelta)
    case 'bulletList':
      return rebaseBulletListNode(node, offsetDelta, lineDelta)
    case 'orderedList':
      return rebaseOrderedListNode(node, offsetDelta, lineDelta)
    case 'taskList':
      return rebaseTaskListNode(node, offsetDelta, lineDelta)
    case 'blankLines':
      return rebaseBlankLinesNode(node, offsetDelta, lineDelta)
  }
}

function rebaseListItemContentItemNode(
  node: ListItemContentItemNode,
  offsetDelta: number,
  lineDelta: number
): ListItemContentItemNode {
  return {
    ...node,
    content: rebaseListItemContentNode(node.content, offsetDelta, lineDelta),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBulletItemNode(
  node: BulletItemNode,
  offsetDelta: number,
  lineDelta: number
): BulletItemNode {
  return {
    ...node,
    content: node.content.map(child => rebaseListItemContentItemNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseOrderedItemNode(
  node: OrderedItemNode,
  offsetDelta: number,
  lineDelta: number
): OrderedItemNode {
  return {
    ...node,
    content: node.content.map(child => rebaseListItemContentItemNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseTaskItemNode(node: TaskItemNode, offsetDelta: number, lineDelta: number): TaskItemNode {
  return {
    ...node,
    content: node.content.map(child => rebaseListItemContentItemNode(child, offsetDelta, lineDelta)),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseListItemBreakNode(
  node: ListItemBreakNode,
  offsetDelta: number,
  lineDelta: number
): ListItemBreakNode {
  return {
    ...node,
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBulletListNode(
  node: BulletListNode,
  offsetDelta: number,
  lineDelta: number
): BulletListNode {
  return {
    ...node,
    content: node.content.map(item =>
      item.type === 'bulletItem'
        ? rebaseBulletItemNode(item, offsetDelta, lineDelta)
        : rebaseListItemBreakNode(item, offsetDelta, lineDelta)
    ),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseOrderedListNode(
  node: OrderedListNode,
  offsetDelta: number,
  lineDelta: number
): OrderedListNode {
  return {
    ...node,
    content: node.content.map(item =>
      item.type === 'orderedItem'
        ? rebaseOrderedItemNode(item, offsetDelta, lineDelta)
        : rebaseListItemBreakNode(item, offsetDelta, lineDelta)
    ),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseTaskListNode(node: TaskListNode, offsetDelta: number, lineDelta: number): TaskListNode {
  return {
    ...node,
    content: node.content.map(item =>
      item.type === 'taskItem'
        ? rebaseTaskItemNode(item, offsetDelta, lineDelta)
        : rebaseListItemBreakNode(item, offsetDelta, lineDelta)
    ),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBlockquoteContentNode(
  node: BlockquoteContentNode,
  offsetDelta: number,
  lineDelta: number
): BlockquoteContentNode {
  switch (node.type) {
    case 'paragraph':
      return rebaseParagraphNode(node, offsetDelta, lineDelta)
    case 'heading':
      return rebaseHeadingNode(node, offsetDelta, lineDelta)
    case 'bulletList':
      return rebaseBulletListNode(node, offsetDelta, lineDelta)
    case 'orderedList':
      return rebaseOrderedListNode(node, offsetDelta, lineDelta)
    case 'taskList':
      return rebaseTaskListNode(node, offsetDelta, lineDelta)
    case 'blockquote':
      return rebaseBlockquoteNode(node, offsetDelta, lineDelta)
    case 'blankLines':
      return rebaseBlankLinesNode(node, offsetDelta, lineDelta)
  }
}

function rebaseBlockquoteItemNode(
  node: BlockquoteItemNode,
  offsetDelta: number,
  lineDelta: number
): BlockquoteItemNode {
  return {
    ...node,
    content: rebaseBlockquoteContentNode(node.content, offsetDelta, lineDelta),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBlockquoteItemBreakNode(
  node: BlockquoteItemBreakNode,
  offsetDelta: number,
  lineDelta: number
): BlockquoteItemBreakNode {
  return {
    ...node,
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBlockquoteNode(
  node: BlockquoteNode,
  offsetDelta: number,
  lineDelta: number
): BlockquoteNode {
  return {
    ...node,
    content: node.content.map(item =>
      item.type === 'blockquoteItem'
        ? rebaseBlockquoteItemNode(item, offsetDelta, lineDelta)
        : rebaseBlockquoteItemBreakNode(item, offsetDelta, lineDelta)
    ),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseBlankLinesNode(
  node: BlankLinesNode,
  offsetDelta: number,
  lineDelta: number
): BlankLinesNode {
  return {
    ...node,
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

function rebaseRootBlockNode(
  node: RootBlockNode,
  offsetDelta: number,
  lineDelta: number
): RootBlockNode {
  switch (node.type) {
    case 'paragraph':
      return rebaseParagraphNode(node, offsetDelta, lineDelta)
    case 'heading':
      return rebaseHeadingNode(node, offsetDelta, lineDelta)
    case 'bulletList':
      return rebaseBulletListNode(node, offsetDelta, lineDelta)
    case 'orderedList':
      return rebaseOrderedListNode(node, offsetDelta, lineDelta)
    case 'taskList':
      return rebaseTaskListNode(node, offsetDelta, lineDelta)
    case 'codeBlock':
      return rebaseCodeBlockNode(node, offsetDelta, lineDelta)
    case 'blockquote':
      return rebaseBlockquoteNode(node, offsetDelta, lineDelta)
    case 'horizontalRule':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies HorizontalRuleNode
    case 'image':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies ImageBlockNode
    case 'table':
      return rebaseTableNode(node, offsetDelta, lineDelta)
    case 'htmlBlock':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies HtmlBlockNode
    case 'definition':
      return {
        ...node,
        position: rebasePosition(node.position, offsetDelta, lineDelta),
      } satisfies DefinitionNode
  }
}

function rebaseDocumentItemNode(
  node: DocumentItemNode,
  offsetDelta: number,
  lineDelta: number
): DocumentItemNode {
  return {
    ...node,
    content:
      node.content.type === 'blankLines'
        ? rebaseBlankLinesNode(node.content, offsetDelta, lineDelta)
        : rebaseRootBlockNode(node.content, offsetDelta, lineDelta),
    position: rebasePosition(node.position, offsetDelta, lineDelta),
  }
}

export function rebaseDocumentPositions(
  doc: DocumentNode,
  cutPoint: number,
  source: string
): DocumentItemNode[] {
  const lineDelta = countNewlines(source, 0, cutPoint)
  return doc.content.map(item => rebaseDocumentItemNode(item, cutPoint, lineDelta))
}