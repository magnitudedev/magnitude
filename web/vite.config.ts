import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "node:path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: [
      {
        find: "@",
        replacement: resolve(__dirname, "src"),
      },
      {
        find: /^@magnitudedev\/sdk$/,
        replacement: resolve(__dirname, "../packages/sdk/src/index.ts"),
      },
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
      "@magnitudedev/client-common",
      "@magnitudedev/generate-id",
    ],
  },
  // `bun run dev` serves the browser app against the development service started by the desktop dev app or `magnitude serve`.
  server: {
    proxy: Object.fromEntries(["/rpc", "/health", "/inference", "/auth"].map(path => [path, { target: process.env.MAGNITUDE_DEV_SERVICE ?? "http://127.0.0.1:11101", changeOrigin: false }])),
  },
  build: {
    outDir: "dist",
    target: "esnext",
  },
});
