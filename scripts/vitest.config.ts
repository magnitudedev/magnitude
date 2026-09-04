import { defineConfig } from "vitest/config"
import cliConfig from "../cli/vitest.config"

export default defineConfig({
  plugins: cliConfig.plugins,
  test: {
    include: ["scripts/dev-pi.test.ts"],
  },
})
