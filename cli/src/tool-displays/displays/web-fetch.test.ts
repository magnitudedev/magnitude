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

const { webFetchDisplay } = await import('./web-fetch')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

describe('webFetchDisplay error rendering', () => {
  test('shows parenthetical error detail when available', () => {
    const element = webFetchDisplay.render({
      state: {
        toolKey: 'webFetch',
        phase: 'error',
        url: 'https://example.com',
        errorDetail: 'Network timeout while fetching URL',
      } as any,
    }) as any

    const text = htmlToText(renderToStaticMarkup(createElement(() => element)))

    expect(text).toContain('· Error (Network timeout while fetching URL)')
  })

  test('shows generic error label when detail is missing', () => {
    const element = webFetchDisplay.render({
      state: {
        toolKey: 'webFetch',
        phase: 'error',
        url: 'https://example.com',
      } as any,
    }) as any

    const text = htmlToText(renderToStaticMarkup(createElement(() => element)))

    expect(text).toContain('· Error')
    expect(text).not.toContain('· Error (')
  })
})
