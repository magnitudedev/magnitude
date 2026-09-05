import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { resolve } from "node:path"
import { afterEach, describe, expect, test, vi } from "vitest"
import {
  DEVELOPMENT_COUNTER_ENVIRONMENT_VARIABLE,
  nextDevelopmentCounter,
  resolveDevelopmentCounter,
} from "./generate-version"

const roots: string[] = []

const temporaryRoot = async (): Promise<string> => {
  const root = await mkdtemp(resolve(tmpdir(), "magnitude-version-"))
  roots.push(root)
  return root
}

afterEach(async () => {
  vi.unstubAllEnvs()
  await Promise.all(roots.splice(0).map((root) =>
    rm(root, { recursive: true, force: true })
  ))
})

describe("ACN revision allocation", () => {
  test("increments the machine-local development counter", async () => {
    const root = await temporaryRoot()
    const path = resolve(root, "development-revision-counter")

    expect(await nextDevelopmentCounter(path)).toBe(1)
    expect(await nextDevelopmentCounter(path)).toBe(2)

    expect(await readFile(path, "utf8")).toBe("2\n")
  })

  test("uses an explicit development counter without changing the machine-local counter", async () => {
    const root = await temporaryRoot()
    const path = resolve(root, "development-revision-counter")
    await writeFile(path, "7\n")

    vi.stubEnv(DEVELOPMENT_COUNTER_ENVIRONMENT_VARIABLE, "42")

    expect(await resolveDevelopmentCounter(undefined, path)).toBe(42)
    expect(await readFile(path, "utf8")).toBe("7\n")
  })

  test.each(["", "-1", "1.5", "not-a-number"])(
    `rejects invalid ${DEVELOPMENT_COUNTER_ENVIRONMENT_VARIABLE} value %j`,
    async (value) => {
      const root = await temporaryRoot()
      const path = resolve(root, "development-revision-counter")

      await expect(resolveDevelopmentCounter(value, path)).rejects.toThrow(
        `${DEVELOPMENT_COUNTER_ENVIRONMENT_VARIABLE} must be a non-negative safe integer`,
      )
    },
  )
})
