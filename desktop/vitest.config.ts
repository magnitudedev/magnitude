import { defineConfig } from "vitest/config"
import { resolve } from "node:path"

export default defineConfig({
  resolve: { alias: { "@": resolve(import.meta.dirname, "../web/src") } },
  test: {
    name: "desktop",
    root: import.meta.dirname,
    include: ["src/**/*.test.{ts,tsx}"],
  },
})
