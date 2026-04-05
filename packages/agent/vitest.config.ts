import { defineConfig } from 'vitest/config'
import type { Plugin } from 'vite'

function rawMdPlugin(): Plugin {
  return {
    name: 'raw-md',
    transform(code, id) {
      if (id.endsWith('.md')) {
        return `export default ${JSON.stringify(code)}`
      }
    },
  }
}

export default defineConfig({
  plugins: [rawMdPlugin()],
  test: {
    pool: 'forks',
    include: ['src/test-harness/__tests__/*.vitest.ts', 'tests/*.vitest.ts', 'tests/**/*.vitest.ts', 'tests/*.test.ts', 'src/inbox/__tests__/*.test.ts'],
    testTimeout: 30_000,
  },
})
