import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { afterEach, describe, expect, test } from "vitest"
import { advanceAcnRevision } from "./advance-acn-revision"
import { nextDevelopmentCounter } from "./generate-version"

const roots: string[] = []

const temporaryRoot = async (): Promise<string> => {
  const root = await mkdtemp(resolve(tmpdir(), "magnitude-version-"))
  roots.push(root)
  return root
}

afterEach(async () => {
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("ACN revision allocation", () => {
  test("advances the checked-in allocation by exactly one", async () => {
    const root = await temporaryRoot()
    const path = resolve(root, "acn-revision.json")
    await writeFile(path, '{"revision":1}\n')

    await advanceAcnRevision(path)

    expect(JSON.parse(await readFile(path, "utf8"))).toEqual({ revision: 2 })
  })

  test("increments the machine-local development counter", async () => {
    const root = await temporaryRoot()
    const path = resolve(root, "development-revision-counter")

    expect(await nextDevelopmentCounter(path)).toBe(1)
    expect(await nextDevelopmentCounter(path)).toBe(2)

    expect(await readFile(path, "utf8")).toBe("2\n")
  })
})
