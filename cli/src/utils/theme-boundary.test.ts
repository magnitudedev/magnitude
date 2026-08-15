import { Glob } from "bun"
import { describe, expect, it } from "vitest"

describe("CLI theme boundary", () => {
  it("keeps feature code free of raw palette scales and appearance branching", async () => {
    const violations: string[] = []
    const files = new Glob("src/**/*.{ts,tsx}").scan({ cwd: import.meta.dir + "/../..", absolute: true })

    for await (const file of files) {
      if (file.endsWith(".test.ts") || file.endsWith(".test.tsx") || file.endsWith("/utils/theme.ts")) continue
      const source = await Bun.file(file).text()
      if (/\b(?:slate|blue|red|green|orange|violet|indigo)\[\d+\]/.test(source)
        || /theme\.mode\b/.test(source)
        || /defaultCliThemes\.(?:light|dark)/.test(source)) {
        violations.push(file)
      }
    }

    expect(violations).toEqual([])
  })
})
