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
    border: '#4c566a',
  }),
}))

mock.module('../../hooks/use-streaming-reveal', () => ({
  useStreamingReveal: (content: string) => ({
    displayedContent: content,
    showCursor: false,
  }),
}))

const { contentDisplay } = await import('./content')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

describe('inline file write display streaming', () => {
  test('hides streaming preview body when matching file is open in viewer but keeps summary row', () => {
    function ToolRender() {
      return contentDisplay.render({
        state: {
          toolKey: 'fileWrite',
          phase: 'streaming',
          path: 'src/a.ts',
          body: 'const x = 1\nconst y = 2',
          charCount: 20,
          lineCount: 2,
        } as any,
        onFileClick: () => {},
      }) as any
    }

    function Harness() {
      return createElement(SelectedFileProvider, { value: { path: 'src/a.ts' } }, createElement(ToolRender))
    }

    const text = htmlToText(renderToStaticMarkup(createElement(Harness)))

    expect(text).toContain('Writing')
    expect(text).toContain('src/a.ts')
    expect(text).toContain('20 chars · 2 lines')
    expect(text).not.toContain('const x = 1')
  })

  test('shows streaming preview body when viewer closes (no selected file)', () => {
    function ToolRender() {
      return contentDisplay.render({
        state: {
          toolKey: 'fileWrite',
          phase: 'streaming',
          path: 'src/a.ts',
          body: 'const x = 1\nconst y = 2',
          charCount: 20,
          lineCount: 2,
        } as any,
        onFileClick: () => {},
      }) as any
    }

    function Harness() {
      return createElement(SelectedFileProvider, { value: null }, createElement(ToolRender))
    }

    const text = htmlToText(renderToStaticMarkup(createElement(Harness)))

    expect(text).toContain('Writing')
    expect(text).toContain('const')
    expect(text).toContain('x =')
    expect(text).toContain('1')
    expect(text).toContain('y =')
    expect(text).toContain('2')
  })
})
