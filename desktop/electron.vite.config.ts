import { defineConfig } from "electron-vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "node:path";
import { readFileSync } from "node:fs";

const acceptanceConfig = process.env.MAGNITUDE_UPDATE_ACCEPTANCE_CONFIG;
const updateConfiguration = acceptanceConfig ? {
  ...JSON.parse(readFileSync(acceptanceConfig, "utf8")), acceptance: true,
} : {
  origin: "https://magnitude.dev",
  storageOrigin: "https://5r3lqtpag4uzvtxd.public.blob.vercel-storage.com",
  keyId: "magnitude-2026-01",
  publicKey: readFileSync(resolve(__dirname, "../packages/release/resources/distribution/magnitude-2026-01.pub.pem"), "utf8"),
  acceptance: false,
};

export default defineConfig({
  main: {
    define: {
      __MAGNITUDE_UPDATE_CONFIGURATION__: JSON.stringify(updateConfiguration),
    },
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
