import React from 'react'
import { beforeEach, describe, expect, mock, test } from 'bun:test'
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
const renderDocumentCalls: Array<any> = []
const streamingHookCalls: Array<any> = []
const blockRendererCalls: Array<any> = []
let streamingHookResult: { blocks: any[]; pendingText: string } = { blocks: [], pendingText: '' }

mock.module('../hooks/use-theme', () => ({
  useTheme: () => theme,
}))

mock.module('../hooks/use-streaming-reveal', () => ({
  useStreamingReveal: (content: string, isStreaming: boolean) => ({
    displayedContent: content,
    isCatchingUp: false,
    showCursor: isStreaming,
  }),
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({
    terminal: { width: rendererWidth },
    clearSelection() {},
  }),
}))

mock.module('../markdown/blocks', () => ({
  renderDocumentToBlocks: (_doc: any, options: any) => {
    renderDocumentCalls.push({ options })
    return []
  },
  renderDocumentItemToBlocks: () => [],
  spansToText: (spans: any[]) => spans.map((s: any) => s.text || '').join(''),
  slugify: (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, ''),
  extractHeadingSlugsFromBlocks: () => [],
}))

mock.module('../markdown/streaming', () => ({
  useStreamingMarkdownCache: (content: string, options: any) => {
    streamingHookCalls.push({ content, options })
    return streamingHookResult
  },
}))

mock.module('../markdown/block-renderer', () => ({
  BlockRenderer: (props: any) => {
    blockRendererCalls.push(props)
    return <block-renderer data-blocks={props.blocks.length} data-show-cursor={String(!!props.showCursor)} />
  },
}))

const markdownContentModule = await import('../markdown/markdown-content')
const artifactReaderPanelModule = await import('./artifact-reader-panel')

function resetState() {
  renderDocumentCalls.length = 0
  streamingHookCalls.length = 0
  blockRendererCalls.length = 0
  streamingHookResult = { blocks: [], pendingText: '' }
}

function render(node: React.ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

beforeEach(() => {
  resetState()
  rendererWidth = 80
})

describe('MarkdownContent Layer 5 - Suite A MarkdownContent width derivation', () => {
  test('MarkdownContent derives codeBlockWidth from terminal width minus 4', () => {
    rendererWidth = 100

    render(<markdownContentModule.MarkdownContent content="| A | B |\n| - | - |\n| 1 | 2 |" />)

    expect(renderDocumentCalls).toHaveLength(1)
    expect(renderDocumentCalls[0]?.options.codeBlockWidth).toBe(96)
  })

  test('MarkdownContent honors explicit codeBlockWidth override', () => {
    rendererWidth = 100

    render(<markdownContentModule.MarkdownContent content="table" codeBlockWidth={72} />)

    expect(renderDocumentCalls).toHaveLength(1)
    expect(renderDocumentCalls[0]?.options.codeBlockWidth).toBe(72)
  })

  test('StreamingMarkdownContent derives codeBlockWidth from terminal width minus 4', () => {
    rendererWidth = 90

    render(<markdownContentModule.StreamingMarkdownContent content="stream" streaming />)

    expect(streamingHookCalls).toHaveLength(1)
    expect(streamingHookCalls[0]?.options.codeBlockWidth).toBe(86)
  })
})

describe('MarkdownContent Layer 5 - Suite B Artifact panel width budget / fallback', () => {
  test('artifact panel uses narrower markdown width budget than chat markdown', () => {
    rendererWidth = 120

    render(<markdownContentModule.MarkdownContent content="demo" />)
    render(
      <artifactReaderPanelModule.ArtifactReaderPanel
        artifactName="demo.md"
        content="demo"
        onClose={() => {}}
      />,
    )

    expect(renderDocumentCalls[0]?.options.codeBlockWidth).toBe(116)
    expect(streamingHookCalls[0]?.options.codeBlockWidth).toBe(110)
  })

  test('artifact panel width budget matches inner chrome contract', () => {
    rendererWidth = 120

    render(
      <artifactReaderPanelModule.ArtifactReaderPanel
        artifactName="demo.md"
        content="demo"
        onClose={() => {}}
      />,
    )

    expect(streamingHookCalls).toHaveLength(1)
    expect(streamingHookCalls[0]?.options.codeBlockWidth).toBe(110)
  })


})

describe('MarkdownContent Layer 5 - Suite C Streaming consumer behavior', () => {
  test('StreamingMarkdownContent renders pendingText raw while fence is incomplete', () => {
    streamingHookResult = {
      blocks: [{ type: 'heading', level: 1, slug: 'title', content: [{ text: 'Title' }], source: { start: 0, end: 7 } }],
      pendingText: '```js\nconst x = 1',
    }

    const html = render(
      <markdownContentModule.StreamingMarkdownContent
        content={'# Title\n\n```js\nconst x = 1'}
        showCursor
      />,
    )

    expect(blockRendererCalls).toHaveLength(1)
    expect(blockRendererCalls[0]?.showCursor).toBe(false)
    expect(html).toContain('```js\nconst x = 1')
    expect(html).toContain('▍')
  })

  test('StreamingMarkdownContent shows cursor on last semantic block when no pendingText', () => {
    streamingHookResult = {
      blocks: [{ type: 'paragraph', content: [{ text: 'Done' }], source: { start: 0, end: 4 } }],
      pendingText: '',
    }

    const html = render(
      <markdownContentModule.StreamingMarkdownContent
        content="Done"
        showCursor
      />,
    )

    expect(blockRendererCalls).toHaveLength(1)
    expect(blockRendererCalls[0]?.showCursor).toBe(true)
    expect(html).not.toContain('```')
  })

  test('MarkdownContent empty content shows standalone cursor fallback', () => {
    const html = render(<markdownContentModule.MarkdownContent content="" showCursor />)

    expect(renderDocumentCalls).toHaveLength(1)
    expect(blockRendererCalls).toHaveLength(1)
    expect(html).toContain('▍')
  })
})