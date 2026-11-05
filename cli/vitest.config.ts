import { defineConfig } from 'vitest/config'
import { defaultServerConditions, type Plugin } from 'vite'

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
  ssr: {
    resolve: {
      // Test fixtures, which client-common exports only under this condition.
      conditions: [...defaultServerConditions, 'magnitude-test'],
    },
  },
  test: {
    name: 'cli',
    root: import.meta.dirname,
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
