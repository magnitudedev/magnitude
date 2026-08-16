import { defineConfig } from 'vitest/config'
import type { Plugin } from 'vite'

const rawMarkdownPlugin = (): Plugin => ({
  name: 'raw-markdown',
  transform(source, id) {
    if (id.endsWith('.md')) {
      return `export default ${JSON.stringify(source)}`
    }
  },
})

export default defineConfig({
  plugins: [rawMarkdownPlugin()],
  test: {
    name: 'cli',
    root: import.meta.dirname,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
