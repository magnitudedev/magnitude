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
  const text = htmlToText(render(<ContextUsageBar tokenEstimate={5000} hardCap={10000} />))
  expect(text).toContain('50% 5k/10k')
})

test('shows used/Unknown and hides percent when used known and max unknown', () => {
  const text = htmlToText(render(<ContextUsageBar tokenEstimate={5000} hardCap={null} />))
  expect(text).toContain('5k/Unknown')
  expect(text).not.toContain('%')
})

test('unknown used preserves existing behavior (parent does not render when used is unknown)', () => {
  const text = htmlToText(render(<ContextUsageBar tokenEstimate={0} hardCap={null} />))
  expect(text).toContain('0/Unknown')
  expect(text).not.toContain('%')
})