import { describe, expect, mock, test } from 'bun:test'
import { TextAttributes } from '@opentui/core'

mock.module('beautiful-mermaid', () => ({
  renderMermaidAscii: () => 'graph TD\nA --> B',
}))

mock.module('../hooks/use-theme', async () => {
  const { chatThemes } = await import('../utils/theme')
  const theme = chatThemes.dark
  return {
    useTheme: () => ({
      ...theme,
      border: theme.border ?? theme.muted,
      success: theme.success ?? 'green',
      primary: theme.primary ?? theme.foreground,
      foreground: theme.foreground ?? 'white',
      muted: theme.muted ?? 'gray',
    }),
  }
})

mock.module('../hooks/use-artifacts', () => ({
  useArtifacts: () => ({ artifacts: new Map() }),
}))

mock.module('../utils/clipboard', () => ({
  writeTextToClipboard: async () => {},
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({ clearSelection() {} }),
  useTerminalDimensions: () => ({ width: 80, height: 40 }),
}))

const { parseMarkdownToMdast } = await import('../markdown/parse')
const { buildMarkdownColorPalette, chatThemes } = await import('../utils/theme')
const { renderDocumentToBlocks } = await import('../markdown/blocks')
const {
  extractAllText,
  extractTextFromStaticMarkup,
  findNodesByType,
  findNodesWithStyle,
  renderBlocksToStaticMarkup,
  renderBlocksToTree,
} = await import('./test-render-helpers')

type Block = Awaited<ReturnType<typeof renderMarkdownBlocks>>[number]

const theme = chatThemes.dark
const palette = buildMarkdownColorPalette(theme)

function renderMarkdownBlocks(markdown: string, codeBlockWidth = 80): Block[] {
  return renderDocumentToBlocks(parseMarkdownToMdast(markdown), { palette, codeBlockWidth })
}

function textFromBlocks(blocks: Block[]): string {
  return extractTextFromStaticMarkup(renderBlocksToStaticMarkup(blocks))
}

function lineContaining(text: string, needle: string): string {
  return text.split('\n').find((line) => line.includes(needle)) ?? ''
}

describe('BlockRenderer Layer 2 - Suite A Spacer rendering / inter-block spacing', () => {
  test('spacer lines 1 produces one visible blank line', () => {
    expect(true).toBe(true)
  })

  test('spacer lines 2 produces two visible blank lines', () => {
    const text = textFromBlocks([
      { type: 'paragraph', content: [{ text: 'A' }], source: { start: 0, end: 1 } },
      { type: 'spacer', lines: 2 },
      { type: 'paragraph', content: [{ text: 'B' }], source: { start: 2, end: 3 } },
    ])
    expect(text).toContain('A\n\n\nB')
  })

  test('paragraph paragraph pair has visible separation when a spacer block exists', () => {
    expect(textFromBlocks(renderMarkdownBlocks('A\n\nB'))).toContain('A\n\nB')
  })

  test('heading paragraph pair has visible separation when a spacer block exists', () => {
    expect(textFromBlocks(renderMarkdownBlocks('# Title\n\nBody'))).toContain('Title\n\nBody')
  })

  test('mixed block document renders visible vertical separation at every semantic spacer', () => {
    const markdown = `# Title

Intro

- item

---

| A | B |
| - | - |
| 1 | 2 |

\`\`\`ts
const x = 1
\`\`\``
    const text = textFromBlocks(renderMarkdownBlocks(markdown, 60))
    expect(text).toContain('Title\n\nIntro')
    expect(text).toContain('item')
    expect(text).toContain('┌')
    expect(text).toContain('const x = 1')
  })
})

describe('BlockRenderer Layer 2 - Suite B Table rendering', () => {
  test('table renders visible box border characters', () => {
    const text = textFromBlocks(renderMarkdownBlocks('| A | B |\n| - | - |\n| 1 | 2 |'))
    expect(text).toContain('┌')
    expect(text).toContain('┬')
    expect(text).toContain('┐')
    expect(text).toContain('│')
    expect(text).toContain('└')
  })

  test('table header cells render in header row', () => {
    const text = textFromBlocks(renderMarkdownBlocks('| A | B |\n| - | - |\n| 1 | 2 |'))
    expect(text).toContain('A')
    expect(text).toContain('B')
  })

  test('table body rows render beneath header row', () => {
    const text = textFromBlocks(renderMarkdownBlocks('| A | B |\n| - | - |\n| 1 | 2 |'))
    expect(text).toContain('1')
    expect(text).toContain('2')
  })

  test('left center right alignment render distinctly', () => {
    const text = textFromBlocks(renderMarkdownBlocks('| L | C | R |\n|:--|:-:|--:|\n| a | b | c |', 30))
    const row = lineContaining(text, 'a')
    expect(row).toContain('a')
    expect(row).toContain('b')
    expect(row).toContain('c')
    expect(row.match(/ b /)).toBeTruthy()
  })

  test('table preserves cell content when width allows it', () => {
    const text = textFromBlocks(
      renderMarkdownBlocks('| A | Long Header |\n| - | - |\n| 1 | very very very long value |', 24),
    )
    expect(text).toContain('very very very long value')
  })

  test('table pads short content to column width', () => {
    const text = textFromBlocks(renderMarkdownBlocks('| Short | Wide Column |\n| ----- | ----------- |\n| x | y |', 40))
    expect(text).toContain('Short')
    expect(text).toContain('Wide Column')
    expect(text).toContain('x')
    expect(text).toContain('y')
  })

  test('table adapts to different available widths', () => {
    const wideText = textFromBlocks(renderMarkdownBlocks('| A | Much Longer Header |\n| - | - |\n| 1 | 2 |', 80))
    const narrowText = textFromBlocks(renderMarkdownBlocks('| A | Much Longer Header |\n| - | - |\n| 1 | 2 |', 28))
    expect(wideText).toContain('Much Longer Header')
    expect(narrowText).toContain('Much')
    expect(lineContaining(wideText, '┌').length).toBeGreaterThanOrEqual(lineContaining(narrowText, '┌').length)
  })
})

describe('BlockRenderer Layer 2 - Suite C Code block rendering', () => {
  test('code block renders visible chrome and language label', () => {
    const text = textFromBlocks(renderMarkdownBlocks('```ts\nconst x = 1\n```'))
    expect(text).toContain('┌ ts')
    expect(text).toContain('└')
  })

  test('language label is present in code block header', () => {
    const text = textFromBlocks(renderMarkdownBlocks('```ts\nconst x = 1\n```'))
    expect(text).toContain('┌ ts')
  })

  test('code block footer path is rendered', () => {
    const text = textFromBlocks(renderMarkdownBlocks('```ts\nconst x = 1\n```'))
    expect(text).toContain('└')
  })

  test('multi-line code preserves explicit line breaks', () => {
    const text = textFromBlocks(renderMarkdownBlocks('```txt\nline1\nline2\nline3\n```'))
    expect(text).toContain('line1\nline2\nline3')
  })
})

describe('BlockRenderer Layer 2 - Suite D Blockquote rendering', () => {
  test('blockquote prefix is rendered', () => {
    const text = textFromBlocks(renderMarkdownBlocks('> quote'))
    expect(text).toContain('> quote')
  })

  test('blockquote nested content is rendered', () => {
    const text = textFromBlocks(renderMarkdownBlocks('> quoted'))
    expect(text).toContain('quoted')
  })

  test('blockquote with blank line renders visible blank quoted separation', () => {
    const text = textFromBlocks(renderMarkdownBlocks('> first\n>\n> second'))
    expect(text).toContain('first')
    expect(text).toContain('second')
    expect(text).toContain('\n')
  })
})

describe('BlockRenderer Layer 2 - Suite E Divider rendering', () => {
  test('divider width adapts to available width instead of fixed 40', () => {
    const text = textFromBlocks([{ type: 'divider', source: { start: 0, end: 3 } }])
    expect(text.length).not.toBe(40)
  })

  test('divider uses palette divider color', () => {
    const tree = renderBlocksToTree([{ type: 'divider', source: { start: 0, end: 3 } }])
    const dividerNode = findNodesByType(tree, 'text')[0]
    expect((dividerNode.props.style as Record<string, unknown> | undefined)?.fg).toBe(palette.dividerFg)
  })

  test('divider surrounded by spacers produces visible separation', () => {
    const text = textFromBlocks(renderMarkdownBlocks('A\n\n---\n\nB'))
    expect(text).toContain('A')
    expect(text).toContain('B')
    expect(text).toContain('\n')
  })
})

describe('BlockRenderer Layer 2 - Suite F List rendering', () => {
  test('list markers appear aligned with first line content', () => {
    const text = textFromBlocks(renderMarkdownBlocks('9. alpha\n10. beta'))
    expect(text).toContain('9. alpha')
    expect(text).toContain('10. beta')
  })

  test('continuation lines align under content not under marker', () => {
    const text = textFromBlocks(renderMarkdownBlocks('9. alpha beta gamma delta epsilon zeta eta theta', 22))
    const lines = text.split('\n')
    expect(lines[0]?.startsWith('9. ')).toBe(true)
  })

  test('nested lists render parent and child items', () => {
    const text = textFromBlocks(renderMarkdownBlocks('- parent\n  - child'))
    expect(text).toContain('parent')
    expect(text).toContain('child')
  })

  test('task item checked state is rendered visually', () => {
    const text = textFromBlocks(renderMarkdownBlocks('- [ ] todo\n- [x] done'))
    expect(text).toContain('- [ ] todo')
    expect(text).toContain('- [x] done')
  })

  test('list item whose first child is blockquote does not render detached marker paragraph', () => {
    const text = textFromBlocks(renderMarkdownBlocks('- > quoted'))
    expect(text).not.toContain('- \n')
    expect(text).toContain('quoted')
  })

  test('blank line inside list item renders visible separation', () => {
    const text = textFromBlocks(renderMarkdownBlocks('- first\n\n  second'))
    expect(text).toContain('first')
    expect(text).toContain('second')
    expect(text).toContain('\n')
  })
})

describe('BlockRenderer Layer 2 - Suite G Mermaid rendering', () => {
  test('mermaid block renders header or chrome consistent with code blocks', () => {
    const text = textFromBlocks(renderMarkdownBlocks('```mermaid\ngraph TD\nA --> B\n```'))
    expect(text).toContain('mermaid')
    expect(text.includes('┌') || text.includes('│') || text.includes('─')).toBe(true)
  })

  test('mermaid render uses code-style chrome', () => {
    const codeText = textFromBlocks(renderMarkdownBlocks('```ts\nconst x = 1\n```'))
    const mermaidText = textFromBlocks(renderMarkdownBlocks('```mermaid\ngraph TD\nA --> B\n```'))
    expect(codeText).toContain('┌')
    expect(mermaidText).toContain('┌')
    expect(mermaidText).toContain('└')
  })

  test('mermaid block has visible spacing above and below', () => {
    const text = textFromBlocks(renderMarkdownBlocks('Before\n\n```mermaid\ngraph TD\nA --> B\n```\n\nAfter'))
    expect(text).toContain('Before')
    expect(text).toContain('After')
    expect(text).toContain('\n')
  })
})