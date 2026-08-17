import { readdirSync, readFileSync } from "node:fs"
import { extname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const sourceRoot = fileURLToPath(new URL("../", import.meta.url))

const sourceFiles = (directory: string): readonly string[] =>
  readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name)
    return entry.isDirectory() ? sourceFiles(path) : [path]
  })

describe("Tailwind styling boundary", () => {
  it("keeps one CSS entrypoint", () => {
    const cssFiles = sourceFiles(sourceRoot)
      .filter((path) => extname(path) === ".css")
      .map((path) => path.slice(sourceRoot.length))

    expect(cssFiles).toEqual(["styles/tailwind.css"])
  })

  it("does not reintroduce the removed semantic CSS variables", () => {
    const source = sourceFiles(sourceRoot)
      .filter((path) => [".ts", ".tsx", ".css"].includes(extname(path)))
      .map((path) => readFileSync(path, "utf8"))
      .join("\n")

    expect(source).not.toMatch(
      /var\(--(?:bg|fg|accent|border|line|diff|syntax|tint)-/
    )
  })

  it("keeps component colors in the Tailwind palette", () => {
    const components = sourceFiles(sourceRoot)
      .filter((path) => extname(path) === ".tsx")
      .map((path) => readFileSync(path, "utf8"))
      .join("\n")

    expect(components).not.toMatch(/#[\da-f]{3,8}\b/i)
    expect(components).not.toMatch(
      /(?:rgb|hsl)a?\((?!\s*0\s*,\s*0\s*,\s*0\s*,)/i
    )
  })

  it("uses Tailwind v4 as the stylesheet entrypoint", () => {
    const stylesheet = readFileSync(
      join(sourceRoot, "styles/tailwind.css"),
      "utf8"
    )

    expect(stylesheet).toContain('@import "tailwindcss" source(none)')
    expect(stylesheet).toContain('@source "../../../desktop/src/renderer.tsx"')
  })
})
