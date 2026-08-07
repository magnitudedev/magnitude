import { defineConfig } from "vitest/config"

export default defineConfig({
  test: {
    name: "@magnitudedev/version",
    include: ["scripts/**/*.test.ts"],
  },
})
