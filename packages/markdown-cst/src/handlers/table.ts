/**
 * Table Handlers (GFM)
 */

import type {
  TableBuilder,
  TableRowBuilder,
  TableCellBuilder,
  TableDelimiterBuilder,
  RawBlockBuilder,
} from '../types'
import type { ParagraphNode, SourcePoint, SourcePosition } from '../schema'
import { finalizeTable } from '../finalize'
import { definePartialHandlers } from './define'
import { isUnsupportedInCurrentContext, addBlockToParent } from './helpers'

function tokenPosition(token: { start: SourcePoint; end: SourcePoint }): SourcePosition {
  return { start: token.start, end: token.end }
}

// =============================================================================
// ENTER HANDLERS
// =============================================================================

export const enter = definePartialHandlers({
  table: (ctx, token) => {
    if (isUnsupportedInCurrentContext(ctx, 'table')) {
      const builder: RawBlockBuilder = {
        builderType: 'rawBlock',
        positionStart: null,
        positionEnd: null,
        startOffset: token.start.offset,
        originalType: 'table',
      }
      ctx.push(builder)
      ctx.enterToken(token, 'rawBlock')
      return
    }

    const builder: TableBuilder = {
      builderType: 'table',
      positionStart: null,
      positionEnd: null,
      rows: [],
      columnAlignments: [],
      columnWidths: [],
      delimiterRow: '',
      pendingLinePrefix: '',
    }
    ctx.push(builder)
    ctx.enterToken(token, 'table')
  },

  // tableHead/tableBody - just containers, no action needed
  tableHead: () => {},
  tableBody: () => {},

  tableRow: (ctx, token) => {
    const table = ctx.find('table')
    if (!table) return

    const builder: TableRowBuilder = {
      builderType: 'tableRow',
      positionStart: null,
      positionEnd: null,
      cells: [],
      rawRow: ctx.slice(token),
      linePrefix: table.pendingLinePrefix,
    }
    table.pendingLinePrefix = ''
    ctx.push(builder)
    ctx.enterToken(token, 'tableRow')
  },

  tableHeader: (ctx, token) => {
    const table = ctx.find('table')
    if (!table) return

    const cellIndex = ctx.find('tableRow')?.cells.length ?? 0
    const alignment = table.columnAlignments[cellIndex] ?? null

    const builder: TableCellBuilder = {
      builderType: 'tableCell',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      alignment,
      leadingWhitespace: '',
      trailingWhitespace: '',
      sawContent: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'tableCell')
  },

  tableData: (ctx, token) => {
    const table = ctx.find('table')
    if (!table) return

    const cellIndex = ctx.find('tableRow')?.cells.length ?? 0
    const alignment = table.columnAlignments[cellIndex] ?? null

    const builder: TableCellBuilder = {
      builderType: 'tableCell',
      positionStart: null,
      positionEnd: null,
      content: [],
      currentText: null,
      pendingSoftBreak: null,
      pendingHardBreak: null,
      alignment,
      leadingWhitespace: '',
      trailingWhitespace: '',
      sawContent: false,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'tableCell')
  },

  tableDelimiterRow: (ctx, token) => {
    const table = ctx.find('table')
    if (!table) return

    table.delimiterRow = table.pendingLinePrefix + ctx.slice(token)
    table.pendingLinePrefix = ''
  },

  tableDelimiter: (ctx, token) => {
    const builder: TableDelimiterBuilder = {
      builderType: 'tableDelimiter',
      positionStart: null,
      positionEnd: null,
      hasLeftColon: false,
      hasRightColon: false,
      fillerWidth: 0,
    }
    ctx.push(builder)
    ctx.enterToken(token, 'tableDelimiter')
  },

  tableContent: (ctx) => {
    const cell = ctx.find('tableCell')
    if (cell) {
      cell.sawContent = true
      cell.trailingWhitespace = ''
    }
  },

  // tableCellDivider: the '|' character - structural separator, not needed in AST
  tableCellDivider: () => {},
  // tableDelimiterMarker: the ':' in delimiter row - alignment captured in exit handler
  tableDelimiterMarker: () => {},
  // tableDelimiterFiller: the '---' in delimiter row - width captured in exit handler
  tableDelimiterFiller: () => {},
})

// =============================================================================
// EXIT HANDLERS
// =============================================================================

export const exit = definePartialHandlers({
  table: (ctx, token) => {
    const rawBuilder = ctx.current('rawBlock')
    if (rawBuilder && rawBuilder.originalType === 'table') {
      ctx.exitToken(token)
      const builder = ctx.pop('rawBlock')
      const text = ctx.source.slice(builder.startOffset, token.end.offset)
      const para: ParagraphNode = {
        type: 'paragraph',
        content: text ? [{ type: 'text', text, position: tokenPosition(token) }] : undefined,
        position: tokenPosition(token),
      }
      addBlockToParent(ctx, para)
      return
    }

    ctx.exitToken(token)
    const builder = ctx.pop('table')
    const node = finalizeTable(builder)
    addBlockToParent(ctx, node)
  },

  tableHead: () => {},
  tableBody: () => {},

  tableRow: (ctx, token) => {
    ctx.exitToken(token)
    const builder = ctx.pop('tableRow')

    const table = ctx.find('table')
    if (table) {
      table.rows.push(builder)
    }
  },

  tableHeader: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('tableCell')

    const row = ctx.find('tableRow')
    if (row) {
      row.cells.push(builder)
    }
  },

  tableData: (ctx, token) => {
    ctx.exitToken(token)
    ctx.flushText()
    const builder = ctx.pop('tableCell')

    const row = ctx.find('tableRow')
    if (row) {
      row.cells.push(builder)
    }
  },

  tableDelimiterRow: () => {},

  tableDelimiter: (ctx, token) => {
    ctx.exitToken(token)
    const builder = ctx.pop('tableDelimiter')

    let alignment: 'left' | 'center' | 'right' | null = null
    if (builder.hasLeftColon && builder.hasRightColon) {
      alignment = 'center'
    } else if (builder.hasRightColon) {
      alignment = 'right'
    } else if (builder.hasLeftColon) {
      alignment = 'left'
    }

    const table = ctx.find('table')
    if (table) {
      table.columnAlignments.push(alignment)
      table.columnWidths.push(builder.fillerWidth)
    }
  },

  tableDelimiterMarker: (ctx, token) => {
    const marker = ctx.slice(token)
    const builder = ctx.find('tableDelimiter')
    if (builder && marker === ':') {
      if (builder.fillerWidth === 0) {
        builder.hasLeftColon = true
      } else {
        builder.hasRightColon = true
      }
    }
  },

  tableDelimiterFiller: (ctx, token) => {
    const builder = ctx.find('tableDelimiter')
    if (builder) {
      builder.fillerWidth = ctx.slice(token).length
    }
  },

  // tableContent: content marking handled in enter, inline handlers process actual content
  tableContent: () => {},
  // tableCellDivider: the '|' character - structural separator, not needed in AST
  tableCellDivider: () => {},
})
