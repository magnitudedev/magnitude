import { expect, test } from 'vitest'

test('uses bottom sticky start only while actively streaming with no section target', async () => {
  const source = await Bun.file(new URL('./file-viewer-panel.tsx', import.meta.url)).text()

  expect(source).toContain(
    "stickyStart={isActivelyStreaming && !scrollToSection ? 'bottom' : 'top'}",
  )
})
