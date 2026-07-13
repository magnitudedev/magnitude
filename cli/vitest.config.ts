import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    name: 'cli',
    root: import.meta.dirname,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
