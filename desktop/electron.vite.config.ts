import { defineConfig } from "electron-vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "node:path";
import { readFileSync } from "node:fs";

export default defineConfig({
  main: {
    plugins: [{ name: "harness-skill-text", load(id) { if (id.endsWith(".md")) return `export default ${JSON.stringify(readFileSync(id, "utf8"))}` } }],
    build: {
      // Workspace packages publish TypeScript source for Bun. Bundle them for
      // Electron's Node runtime so production does not depend on repository
      // source files or Node's TypeScript resolution behavior.
      externalizeDeps: false,
      rollupOptions: {
        input: {
          main: resolve(__dirname, "src/main.ts"),
        },
      },
    },
  },
  preload: {
    build: {
      externalizeDeps: false,
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
          find: "@",
          replacement: resolve(__dirname, "../web/src"),
        },
        {
          find: "@magnitudedev/web",
          replacement: resolve(__dirname, "../web/src/index.tsx"),
        },
        {
          find: /^@magnitudedev\/sdk$/,
          replacement: resolve(__dirname, "../packages/sdk/src/index.ts"),
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
          "@magnitudedev/sdk/desktop-host",
          "@magnitudedev/daemon-management",
          "@magnitudedev/daemon-management/desktop-native",
          "@magnitudedev/utils",
          "@magnitudedev/harness-connections",
          "@magnitudedev/daemon-management/node",
          "@magnitudedev/storage",
          "@magnitudedev/release",
        "@magnitudedev/client-common",
        "@magnitudedev/generate-id",
        "@magnitudedev/web",
      ],
    },
  },
});
