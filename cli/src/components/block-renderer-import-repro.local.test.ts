import { test, expect, mock } from 'bun:test'

mock.module('../hooks/use-theme', async () => {
  const { defaultCliThemes } = await import('../utils/theme')
  return { useTheme: () => defaultCliThemes.dark }
})

mock.module('../utils/clipboard', () => ({
  writeTextToClipboard: async () => {},
}))

mock.module('@opentui/react', () => ({
  useRenderer: () => ({ clearSelection() {} }),
  useTerminalDimensions: () => ({ width: 80, height: 24 }),
}))

import './test-render-helpers'

test('import repro', () => {
  expect(true).toBe(true)
})
