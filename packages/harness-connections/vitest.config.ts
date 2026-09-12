import { defineConfig } from "vitest/config"
export default defineConfig({
  plugins: [{ name: "raw-markdown", transform(source, id) { if (id.endsWith(".md")) return `export default ${JSON.stringify(source)}` } }],
  test: { name: "harness-connections", root: import.meta.dirname, include: ["src/**/*.test.ts"] },
})
