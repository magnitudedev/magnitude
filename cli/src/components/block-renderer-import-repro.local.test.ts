import { test, expect, mock } from 'bun:test'

mock.module('../hooks/use-theme', () => ({
  useTheme: () => ({
    foreground: 'white',
    muted: 'gray',
    primary: 'blue',
    success: 'green',
    border: 'gray',
    link: 'blue',
    syntax: {},
  }),
}))

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
