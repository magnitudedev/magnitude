import { describe, expect, test } from 'bun:test'
import { parseMarkdownToMdast } from '../markdown/parse'
import { renderDocumentToBlocks } from '../markdown/blocks'
import { buildMarkdownColorPalette, chatThemes } from '../utils/theme'

describe('block timeout repro', () => {
  test('parse + blocks', () => {
    const doc = parseMarkdownToMdast('A')
    const palette = buildMarkdownColorPalette(chatThemes.dark)
    const blocks = renderDocumentToBlocks(doc, { palette })
    expect(blocks.length).toBeGreaterThanOrEqual(1)
  })
})
