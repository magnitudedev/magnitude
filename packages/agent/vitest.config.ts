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

function rawJinjaPlugin(): Plugin {
  return {
    name: 'raw-jinja',
    transform(code, id) {
      if (id.endsWith('.jinja')) {
        return `export default ${JSON.stringify(code)}`
      }
    },
  }
}

export default defineConfig({
  plugins: [rawMdPlugin(), rawJinjaPlugin()],
  test: {
    pool: 'forks',
    include: ['src/test-harness/__tests__/*.vitest.ts', 'tests/*.vitest.ts', 'tests/**/*.vitest.ts', 'tests/*.test.ts', 'src/inbox/__tests__/*.test.ts', 'src/runtime/__tests__/*.test.ts'],
    testTimeout: 30_000,
  },
})
