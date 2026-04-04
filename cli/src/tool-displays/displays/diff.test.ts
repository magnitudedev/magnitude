import { describe, expect, mock, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { SelectedFileProvider } from '../../hooks/use-file-viewer'

mock.module('../../hooks/use-theme', () => ({
  useTheme: () => ({
    info: '#88c0d0',
    foreground: '#eceff4',
    primary: '#81a1c1',
    syntax: { string: '#a3be8c' },
    secondary: '#4c566a',
    error: '#bf616a',
    muted: '#616e88',
    link: '#8fbcbb',
  }),
}))

mock.module('../../hooks/use-streaming-reveal', () => ({
  useStreamingReveal: (content: string) => ({
    displayedContent: content,
    showCursor: false,
  }),
}))

const { diffDisplay } = await import('./diff')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

describe('inline diff display streaming', () => {
  test('summary reflects streaming state', () => {
    expect(diffDisplay.summary({
      toolKey: 'fileEdit',
      phase: 'streaming',
      path: 'a.ts',
      oldText: 'old',
      newText: 'new',
      replaceAll: false,
      streamingTarget: 'new',
      baseContent: null,
      diffs: [],
    })).toBe('Editing a.ts')
  })

  test('summary reflects completed state', () => {
    expect(diffDisplay.summary({
      toolKey: 'fileEdit',
      phase: 'completed',
      path: 'a.ts',
      oldText: 'old',
      newText: 'new',
      replaceAll: false,
      streamingTarget: null,
      baseContent: null,
      diffs: [],
    })).toBe('Edited a.ts')
  })

  test('renders surrounding context from state.diffs while streaming inline edit', () => {
    function Harness() {
      return createElement(SelectedFileProvider, { value: null },
        diffDisplay.render({
          state: {
            toolKey: 'fileEdit',
            phase: 'streaming',
            path: 'src/a.ts',
            oldText: 'const before = 1',
            newText: 'const after = 2',
            replaceAll: false,
            streamingTarget: 'new',
            baseContent: null,
            diffs: [
              {
                contextBefore: ['function demo() {', '  // keep this'],
                removedLines: ['  const before = 1'],
                addedLines: ['  const after = 2'],
                contextAfter: ['  return after', '}'],
              },
            ],
          },
          isExpanded: false,
          onToggle: () => {},
          onFileClick: () => {},
        }) as any,
      )
    }

    const text = htmlToText(renderToStaticMarkup(createElement(Harness)))
    expect(text).toContain('Editing')
    expect(text).toContain('function demo() {')
    expect(text).toContain('// keep this')
    expect(text).toContain('const before = 1')
    expect(text).toContain('const after = 2')
    expect(text).toContain('return after')
  })

  test('hides streaming diff body when matching file is open in viewer while keeping header', () => {
    function ToolRender() {
      return diffDisplay.render({
        state: {
          toolKey: 'fileEdit',
          phase: 'streaming',
          path: 'src/a.ts',
          oldText: 'const before = 1',
          newText: 'const after = 2',
          replaceAll: false,
          streamingTarget: 'new',
          baseContent: null,
          diffs: [
            {
              contextBefore: ['function demo() {'],
              removedLines: ['  const before = 1'],
              addedLines: ['  const after = 2'],
              contextAfter: ['}'],
            },
          ],
        } as any,
        isExpanded: false,
        onToggle: () => {},
        onFileClick: () => {},
      }) as any
    }

    function Harness() {
      return createElement(SelectedFileProvider, { value: { path: 'src/a.ts' } }, createElement(ToolRender))
    }

    const text = htmlToText(renderToStaticMarkup(createElement(Harness)))

    expect(text).toContain('Editing')
    expect(text).toContain('src/a.ts')
    expect(text).not.toContain('const before = 1')
    expect(text).not.toContain('const after = 2')
  })
})
