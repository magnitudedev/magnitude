import { defineConfig } from 'vitest/config'

export default defineConfig({
  // Harness connections import their skill document as text, as Bun does at compile time.
  plugins: [{ name: 'raw-markdown', transform(source, id) { if (id.endsWith('.md')) return `export default ${JSON.stringify(source)}` } }],
  test: {
    include: ['src/**/*.test.ts'],
  },
})
