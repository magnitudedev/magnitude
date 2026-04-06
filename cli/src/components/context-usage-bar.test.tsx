import React from 'react'
import { test, expect, mock } from 'bun:test'
import { renderToStaticMarkup } from 'react-dom/server'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    muted: '#888888',
  }),
}))

const { ContextUsageBar } = await import('./context-usage-bar')

const htmlToText = (html: string): string => html.replace(/<[^>]*>/g, '').replace(/\s+/g, ' ').trim()

function render(node: React.ReactNode) {
  return renderToStaticMarkup(<>{node}</>)
}

test('shows used/max and percent when used and max are known', () => {
  const text = htmlToText(render(<ContextUsageBar contextTokens={5000} hardCap={10000} />))
  expect(text).toContain('50% 5k/10k')
})

test('shows used/Unknown and hides percent when used known and max unknown', () => {
  const text = htmlToText(render(<ContextUsageBar contextTokens={5000} hardCap={null} />))
  expect(text).toContain('5k/Unknown')
  expect(text).not.toContain('%')
})

test('shows -/full-window and hides percent when current context is zero', () => {
  const text = htmlToText(render(<ContextUsageBar contextTokens={0} hardCap={10000} />))
  expect(text).toContain('-/10k')
  expect(text).not.toContain('%')
})

test('shows -/Unknown and hides percent when current context is zero and max unknown', () => {
  const text = htmlToText(render(<ContextUsageBar contextTokens={0} hardCap={null} />))
  expect(text).toContain('-/Unknown')
  expect(text).not.toContain('%')
})