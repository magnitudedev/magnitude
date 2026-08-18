import { defineConfig } from "vitest/config"
import { resolve } from "node:path"

export default defineConfig({
  resolve: {
    alias: [
      { find: "@", replacement: resolve(__dirname, "src") },
      { find: /^@magnitudedev\/sdk$/, replacement: resolve(__dirname, "../packages/sdk/src/browser.ts") },
      { find: "@magnitudedev/client-common", replacement: resolve(__dirname, "../packages/client-common/src/index.ts") },
    ],
  },
  test: {
    name: "web",
    root: import.meta.dirname,
    include: ["src/**/*.test.{ts,tsx}"],
  },
})
