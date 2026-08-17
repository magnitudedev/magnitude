import { defineConfig } from "electron-vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"
import { resolve } from "node:path"

export default defineConfig({
  main: {
    build: {
      rollupOptions: {
        // sqlite3 is a native CommonJS addon. It must remain a runtime Node
        // dependency; bundling its tracing helper into ESM erases __filename.
        external: ["sqlite3"],
        input: {
          main: resolve(__dirname, "src/main.ts"),
        },
      },
    },
  },
  preload: {
    build: {
      rollupOptions: {
        input: {
          preload: resolve(__dirname, "src/preload.ts"),
        },
      },
    },
  },
  renderer: {
    root: ".",
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, "index.html"),
        },
      },
    },
    resolve: {
      alias: [
        {
          find: "@magnitudedev/web",
          replacement: resolve(__dirname, "../web/src/index.tsx"),
        },
        {
          find: /^@magnitudedev\/sdk$/,
          replacement: resolve(__dirname, "../packages/sdk/src/browser.ts"),
        },
        {
          find: "@web-styles",
          replacement: resolve(__dirname, "../web/src/styles"),
        },
      ],
    },
    server: {
      fs: {
        allow: [resolve(__dirname, "..")],
      },
    },
    plugins: [react(), tailwindcss()],
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
        "@magnitudedev/client-common",
        "@magnitudedev/generate-id",
        "@magnitudedev/web",
      ],
    },
  },
})
