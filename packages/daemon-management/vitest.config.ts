import { defineConfig } from "vitest/config"

export default defineConfig({ test: { name: "daemon-management", include: ["src/**/*.test.ts"] } })
