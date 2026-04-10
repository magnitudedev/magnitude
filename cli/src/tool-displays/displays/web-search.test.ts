import { describe, expect, mock, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

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

const { webSearchDisplay } = await import('./web-search')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

describe('webSearchDisplay error rendering', () => {
  test('shows parenthetical error detail when available', () => {
    const element = webSearchDisplay.render({
      state: {
        toolKey: 'webSearch',
        phase: 'error',
        query: 'magnitude search',
        sources: [],
        errorDetail:
          'Current provider does not support web search. To enable web search, set MAGNITUDE_SEARCH_PROVIDER to one of: anthropic, openai, gemini, openrouter, vercel, github-copilot.',
      } as any,
      isExpanded: false,
      onToggle: () => {},
    }) as any

    const text = htmlToText(renderToStaticMarkup(createElement(() => element)))

    expect(text).toContain('· Error (Current provider does not support web search.')
    expect(text).toContain('To enable web search, set MAGNITU…')
  })

  test('shows generic error label when detail is missing', () => {
    const element = webSearchDisplay.render({
      state: {
        toolKey: 'webSearch',
        phase: 'error',
        query: 'magnitude search',
        sources: [],
      } as any,
      isExpanded: false,
      onToggle: () => {},
    }) as any

    const text = htmlToText(renderToStaticMarkup(createElement(() => element)))

    expect(text).toContain('· Error')
    expect(text).not.toContain('· Error (')
  })
})
