import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: 'client-common',
    root: import.meta.dirname,
    include: ['src/**/*.test.ts'],
    exclude: ['src/utils/strings.display-width.test.ts'],
  },
})
