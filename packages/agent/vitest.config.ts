import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['src/test-harness/__tests__/*.vitest.ts', 'tests/*.vitest.ts', 'tests/**/*.vitest.ts', 'tests/*.test.ts', 'src/inbox/__tests__/*.test.ts'],
    testTimeout: 30_000,
  },
})
