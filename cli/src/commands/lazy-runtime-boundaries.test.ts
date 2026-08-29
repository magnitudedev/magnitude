import { describe, expect, it } from "vitest"

const commandFamilies = [
  ["connections", "connections-runtime"],
  ["inference", "inference-runtime"],
  ["interactive", "interactive-command-runtime"],
  ["server", "server-runtime"],
  ["update", "update-runtime"],
] as const

describe("CLI command runtime boundaries", () => {
  for (const [family, runtime] of commandFamilies) {
    it(`${family} keeps its runtime behind one lazy boundary`, async () => {
      const source = await Bun.file(new URL(`./${family}.ts`, import.meta.url)).text()
      const valueImports = source.match(/^import(?!\s+type\b).*$/gm) ?? []
      const runtimeImports = source.match(
        new RegExp(`import\\(\\"\\./${runtime}\\"\\)`, "g"),
      ) ?? []

      expect(valueImports).toEqual([])
      expect(runtimeImports).toHaveLength(1)
    })
  }
})
