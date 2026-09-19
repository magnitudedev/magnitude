import { readFileSync } from "node:fs"
import { expect, test } from "vitest"

test("harness installation is portable to a clean worker without developer filesystem links", () => {
  const root = new URL("../tools/", import.meta.url)
  const manifest = JSON.parse(readFileSync(new URL("package.json", root), "utf8"))
  const lock = JSON.parse(readFileSync(new URL("package-lock.json", root), "utf8"))
  expect(lock.packages[""].dependencies).toEqual(manifest.dependencies)
  for (const [path, item] of Object.entries(lock.packages) as [string, { link?: boolean; resolved?: string }][]) {
    if (path === "") continue
    expect(path.startsWith("node_modules/"), path).toBe(true)
    expect(item.link, path).not.toBe(true)
    if (item.resolved) expect(item.resolved.startsWith("https://registry.npmjs.org/"), path).toBe(true)
  }
  for (const [name, version] of Object.entries(manifest.dependencies)) {
    expect(lock.packages[`node_modules/${name}`].version).toBe(version)
  }
})
