import { describe, expect, it } from "vitest"
import { RelativePathSchema } from "@magnitudedev/sdk"
import {
  canMoveProjectEntryToDirectory,
  isProjectPathWithin,
  parentProjectPath,
  translateProjectPath,
} from "./project-file-paths"

const path = RelativePathSchema.make

describe("project file paths", () => {
  it("finds root and nested parents", () => {
    expect(parentProjectPath(path("README.md"))).toBe("")
    expect(parentProjectPath(path("src/components/button.tsx"))).toBe("src/components")
  })

  it("distinguishes descendants without matching path prefixes", () => {
    expect(isProjectPathWithin(path("src"), path("src"))).toBe(true)
    expect(isProjectPathWithin(path("src/components/button.tsx"), path("src"))).toBe(true)
    expect(isProjectPathWithin(path("src-old/index.ts"), path("src"))).toBe(false)
    expect(isProjectPathWithin(path("README.md"), path(""))).toBe(true)
  })

  it("translates an entry and every descendant after a move", () => {
    expect(translateProjectPath(path("src"), path("src"), path("packages/src"))).toBe("packages/src")
    expect(translateProjectPath(path("src/components/button.tsx"), path("src"), path("packages/src")))
      .toBe("packages/src/components/button.tsx")
    expect(translateProjectPath(path("src-old/index.ts"), path("src"), path("packages/src")))
      .toBe("src-old/index.ts")
  })

  it("allows only meaningful, non-recursive filesystem moves", () => {
    const file = { path: path("src/index.ts"), kind: "file" as const }
    const directory = { path: path("src/components"), kind: "directory" as const }
    expect(canMoveProjectEntryToDirectory(file, path(""))).toBe(true)
    expect(canMoveProjectEntryToDirectory(file, path("src"))).toBe(false)
    expect(canMoveProjectEntryToDirectory(directory, path("packages"))).toBe(true)
    expect(canMoveProjectEntryToDirectory(directory, path("src/components/nested"))).toBe(false)
    expect(canMoveProjectEntryToDirectory(directory, path("src"))).toBe(false)
  })
})
