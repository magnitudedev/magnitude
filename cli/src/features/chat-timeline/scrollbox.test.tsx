import { expect, test } from 'vitest'

test('root chat scrollbox leaves sticky bottom and history reachability to the shared controller', async () => {
  const source = await Bun.file(new URL('./scrollbox.tsx', import.meta.url)).text()

  expect(source).not.toContain('stickyScroll')
  expect(source).not.toContain('stickyStart')
  expect(source).toContain("justifyContent: hasMoreBefore ? 'flex-start' : 'flex-end'")
})
