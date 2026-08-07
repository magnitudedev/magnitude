import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    name: "@magnitudedev/acn-protocol",
    include: ["src/**/*.test.ts"],
  },
})
