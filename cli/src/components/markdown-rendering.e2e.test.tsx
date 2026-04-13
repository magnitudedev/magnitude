import React from 'react'
import { describe, expect, mock, test } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'
import { chatThemes } from '../utils/theme'

const theme = {
  ...chatThemes.dark,
  border: chatThemes.dark.border ?? chatThemes.dark.muted,
  success: chatThemes.dark.success ?? 'green',
  error: chatThemes.dark.error ?? 'red',
  info: chatThemes.dark.info ?? 'blue',
  warning: chatThemes.dark.warning ?? 'yellow',
  foreground: chatThemes.dark.foreground ?? 'white',
  muted: chatThemes.dark.muted ?? 'gray',
}

let rendererWidth = 80

mock.module('../hooks/use-theme', () => ({
  useTheme: () => theme,
}))

mock.module('../hooks/use-artifacts', () => ({
  useArtifacts: () => ({ artifacts: new Map() }),
}))

mock.module('../utils/clipboard', () => ({
  writeTextToClipboard: async () => {},
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({
    terminal: { width: rendererWidth },
    clearSelection() {},
  }),
  useTerminalDimensions: () => ({
    width: rendererWidth,
    height: 40,
  }),
}))

mock.module('../hooks/use-local-width', () => ({
  useLocalWidth: () => ({ ref: { current: null }, onSizeChange: () => {}, width: rendererWidth - 4 }),
}))

const { extractTextFromStaticMarkup } = await import('./test-render-helpers')
const { MarkdownContent, StreamingMarkdownContent } = await import('../markdown/markdown-content')
const { ArtifactReaderPanel } = await import('./artifact-reader-panel')

function renderText(node: React.ReactNode): string {
  return extractTextFromStaticMarkup(renderToStaticMarkup(<div>{node}</div>))
}

describe('Markdown rendering Layer 6 - Suite A Complex mixed documents', () => {
  test('complex mixed document renders readable spacing across all block types', () => {
    rendererWidth = 100
    const markdown = `# Title

Intro paragraph.

- item 1
- item 2

---

| A | B |
| - | - |
| 1 | 2 |

\`\`\`ts
const x = 1
\`\`\`

> quoted

\`\`\`mermaid
graph TD
A --> B
\`\`\``

    const text = renderText(<MarkdownContent content={markdown} />)

    expect(text).toContain('Title')
    expect(text).toContain('Intro paragraph.')
    expect(text).toContain('item 1')
    expect(text).toContain('item 2')
    expect(text).toContain('────────────────')
    expect(text).toContain('┌')
    expect(text).toContain('┬')
    expect(text).toContain('┐')
    expect(text).toContain('const x = 1')
    expect(text).toContain('> quoted')
    expect(text).toContain('│A│')
    expect(text).toContain('│B│')
    expect(text.indexOf('Title')).toBeLessThan(text.indexOf('Intro paragraph.'))
    expect(text.indexOf('Intro paragraph.')).toBeLessThan(text.indexOf('item 1'))
    expect(text.indexOf('item 2')).toBeLessThan(text.indexOf('const x = 1'))
  })

  test('nested structures render without collapsing', () => {
    const markdown = `> - parent
>   - child
>
> paragraph`
    const text = renderText(<MarkdownContent content={markdown} />)

    expect(text).toContain('> - parent')
    expect(text).toContain('- child')
    expect(text).toContain('paragraph')
    expect(text.indexOf('parent')).toBeLessThan(text.indexOf('child'))
    expect(text.indexOf('child')).toBeLessThan(text.indexOf('paragraph'))
  })

  test('blockquote inside list item renders as one coherent item', () => {
    const markdown = `- intro

  > quoted inside item`
    const text = renderText(<MarkdownContent content={markdown} />)

    expect(text).toContain('intro')
    expect(text).toContain('quoted inside item')
    expect(text).not.toContain('- \n')
  })
})

describe('Markdown rendering Layer 6 - Suite B Tables in multiple width contexts', () => {
  test('same table renders differently in normal markdown view than in artifact panel', () => {
    rendererWidth = 100
    const markdown = `| A | Much Longer Header |
| - | - |
| 1 | very very long value |`

    const normalText = renderText(<MarkdownContent content={markdown} />)
    const panelText = renderText(
      <ArtifactReaderPanel
        artifactName="table.md"
        content={markdown}
        onClose={() => {}}
      />,
    )

    expect(normalText).toContain('Much Longer Header')
    expect(panelText).toContain('table.md')
    expect(panelText).toContain('Much Longer Header')
    expect(normalText).not.toBe(panelText)
  })

  test('table header text and alignment survive full pipeline', () => {
    const text = renderText(
      <MarkdownContent content={'| Left | Center | Right |\n|:-----|:------:|------:|\n| aa | bb | cc |'} codeBlockWidth={30} />,
    )

    expect(text).toContain('Left')
    expect(text).toContain('Center')
    expect(text).toContain('Right')
    expect(text).toContain('aa')
    expect(text).toContain('bb')
    expect(text).toContain('cc')
    expect(text.indexOf('Left')).toBeLessThan(text.indexOf('Center'))
    expect(text.indexOf('Center')).toBeLessThan(text.indexOf('Right'))
  })
})

describe('Markdown rendering Layer 6 - Suite C Streaming end-to-end', () => {
  test('streaming progressive rendering reaches the same final output as non-streaming render', () => {
    const markdown = `# Title

Body

- item

\`\`\`ts
const x = 1
\`\`\``
    const prefixes = ['#', '# Title\n\nBo', '# Title\n\nBody\n\n- item', markdown]

    for (const prefix of prefixes) {
      renderText(<StreamingMarkdownContent content={prefix} streaming />)
    }

    const finalStreaming = renderText(<StreamingMarkdownContent content={markdown} streaming />)
    const finalStatic = renderText(<MarkdownContent content={markdown} />)
    expect(finalStreaming).toBe(finalStatic)
  })

  test('streaming fenced code shows raw pending text then upgrades to code chrome on close', () => {
    const incomplete = 'Text\n\n```py\nprint(1)'
    const complete = 'Text\n\n```py\nprint(1)\n```'

    const pendingText = renderText(<StreamingMarkdownContent content={incomplete} streaming showCursor />)
    const completeText = renderText(<StreamingMarkdownContent content={complete} streaming showCursor />)

    expect(pendingText).toContain('```py\nprint(1)')
    expect(pendingText).not.toContain('┌ py')
    expect(completeText).toContain('┌ py')
    expect(completeText).not.toContain('```py\nprint(1)')
  })

  test('streaming preserves block ordering and spacers through append sequence', () => {
    const steps = [
      '# T',
      '# T\n\nA',
      '# T\n\nA\n\n- x',
      '# T\n\nA\n\n- x\n\n---\n\n| A | B |\n| - | - |\n| 1 | 2 |',
    ]

    const renders = steps.map((content) => renderText(<StreamingMarkdownContent content={content} streaming />))
    expect(renders[0]).toContain('T')
    expect(renders[1]).toContain('T\n\nA')
    expect(renders[2]).toContain('- x')
    expect(renders[3]).toContain('┌')
    expect(renders[3].indexOf('T')).toBeLessThan(renders[3].indexOf('A'))
    expect(renders[3].indexOf('A')).toBeLessThan(renders[3].indexOf('- x'))
  })
})