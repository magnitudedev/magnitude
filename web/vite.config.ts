import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import { resolve } from "node:path"

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: [
      { find: "@magnitudedev/web", replacement: resolve(__dirname, "src/index.tsx") },
      { find: /^@magnitudedev\/sdk$/, replacement: resolve(__dirname, "../packages/sdk/src/browser.ts") },
    ],
  },
  define: {
    "process.platform": JSON.stringify("browser"),
    "process.arch": JSON.stringify("browser"),
    "process.pid": "0",
    "process.env": "{}",
    "process.versions": "{}",
  },
  optimizeDeps: {
    exclude: [
      "@magnitudedev/sdk",
      "@magnitudedev/protocol",
      "@magnitudedev/client-common",
      "@magnitudedev/generate-id",
    ],
  },
  build: {
    outDir: "dist",
    target: "esnext",
  },
})
