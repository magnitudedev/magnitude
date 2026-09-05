import { defineConfig } from "vitest/config"

export default defineConfig({ test: { name: "release", include: ["src/**/*.test.ts", "scripts/**/*.test.ts"] } })
